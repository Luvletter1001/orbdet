# Copyright (c) OpenMMLab. All rights reserved.
"""OrbdetGDADetector: OrbdetV02Detector + GDA probe loss wiring (Plan-B).

The detector adds nothing to the optimization of the baseline branches.
It only:

* stashes the rotated view's (rot, gt_instances) via a ``rotate_crop``
  override (no copy of the parent's ``loss`` body);
* reconstructs the flipped view from the cropped ori gt (vertical flip,
  bids offset by +0.6 exactly as ``H2RBoxV2Detector.loss`` does);
* in ``_add_auxiliary_losses`` runs the same per-view target assignment as
  the baseline head, compacts probe rows per physical object across the
  three views (positional correspondence via the deterministic ``bid``
  arithmetic), and delegates the tensor math to
  :class:`OrbdetGDAProbeLoss`.

Loss keys contain 'loss' (summed by mmdet's parse_losses); diagnostics do
not (logger-only), mirroring the OrbdetV02Detector convention.
"""
import copy
from typing import List, Tuple

import torch
from mmdet.utils import InstanceList
from torch import Tensor

from mmrotate.models.utils.gda_symmetric_space import envelope_from_wh_theta
from mmrotate.registry import MODELS
from .orbdet_v0_2 import OrbdetV02Detector


def compact_probe_by_object(
        rows_v: List[Tensor], bids_v: List[Tensor],
        extras_v: List[Tensor]
) -> Tuple[Tensor, Tensor, List[int]]:
    """Group per-point tensors by physical object across views.

    Args:
        rows_v: per view, (P_v, C) positive probe rows.
        bids_v: per view, (P_v,) float bids of the assigned gt
            (integer part identifies the object within the view).
        extras_v: per view, (P_v, E) extra per-point tensors (envelope
            target, sin2) -- mean-pooled like the rows (they are
            point-independent for a given object).

    Returns:
        rows3 (M, 3, C): mean-pooled probe rows (views ordered ori, rot,
            flp); extras3 (M, 3, E); keep list of object keys.

    Only objects with at least one positive point in *every* view are
    kept (mirrors the baseline's bcnt == 3 mask).

    Note: the integer part of ``bid`` is globally unique per
    (view, image, gt) because the parent accumulates the offset across
    views -- the *same* physical object has *different* bid integers in
    different views.  Object identity across views is therefore the rank
    of the bid integer within the view's sorted unique bids (gt order is
    preserved across views by the parent's deepcopy/flip).

    Fully vectorized: per view one ``torch.unique`` + one
    ``searchsorted`` + one ``index_reduce_`` (mirrors the baseline's
    own compaction idiom), no per-object Python/kernel loop.
    """
    n_views = len(rows_v)
    pooled_r, pooled_e, n_obj_v = [], [], []
    for v in range(n_views):
        uniq = torch.unique(bids_v[v].long())
        n_obj = uniq.numel()
        n_obj_v.append(n_obj)
        if n_obj == 0:
            pooled_r.append(rows_v[v].new_zeros((0, rows_v[v].size(1))))
            pooled_e.append(extras_v[v].new_zeros((0, extras_v[v].size(1))))
            continue
        rank = torch.searchsorted(uniq, bids_v[v].long())
        pr = rows_v[v].new_zeros((n_obj, rows_v[v].size(1)))
        pr.index_reduce_(0, rank, rows_v[v], 'mean', include_self=False)
        pe = extras_v[v].new_zeros((n_obj, extras_v[v].size(1)))
        pe.index_reduce_(0, rank, extras_v[v], 'mean', include_self=False)
        pooled_r.append(pr)
        pooled_e.append(pe)
    common = sorted(set.intersection(*[set(range(n)) for n in n_obj_v]))
    c = rows_v[0].size(1)
    e = extras_v[0].size(1)
    if not common:
        return (rows_v[0].new_zeros((0, n_views, c)),
                extras_v[0].new_zeros((0, n_views, e)), [])
    idx = torch.tensor(common, device=rows_v[0].device)
    rows3 = torch.stack([pooled_r[v][idx] for v in range(n_views)], dim=1)
    ext3 = torch.stack([pooled_e[v][idx] for v in range(n_views)], dim=1)
    return rows3, ext3, common


@MODELS.register_module()
class OrbdetGDADetector(OrbdetV02Detector):
    """Orbdet-v0.2 detector with the GDA probe objective attached."""

    def rotate_crop(self, batch_inputs, rot=0., size=(768, 768),
                    batch_gt_instances=None, padding='reflection'):
        out = super().rotate_crop(batch_inputs, rot, size,
                                  batch_gt_instances, padding)
        # Stash only the rotated view (rot != 0).  The parent assigns
        # bids (+0.4) to these very gt objects right after this call.
        is_rot_view = isinstance(rot, Tensor) or rot != 0.
        if self.training and batch_gt_instances is not None and is_rot_view:
            self._gda_rot_view = dict(rot=rot, gts=out[1])
        return out

    # -- helpers ---------------------------------------------------------

    def _rebuild_flipped_view(self, gt_ori: InstanceList,
                              gt_rot: InstanceList) -> InstanceList:
        """Reconstruct the flipped view exactly as the parent builds it.

        Parent (H2RBoxV2Detector.loss): vertical flip of the cropped ori
        gt, bids continue the global counter with a +0.6 fraction.
        """
        gt_flp = copy.deepcopy(gt_ori)
        offset = 1 + sum(len(g.bboxes) for g in gt_ori) \
            + sum(len(g.bboxes) for g in gt_rot)
        device = gt_ori[0].bboxes.device if len(gt_ori) else 'cpu'
        for g in gt_flp:
            g.bboxes.flip_(self.crop_size, 'vertical')
            n = len(g.bboxes)
            g.bid = torch.arange(0, n, 1, device=device) + offset + 0.6
            offset += n
        return gt_flp

    def _view_tensors(self, head, stash: List[Tensor], points: List[Tensor],
                      gts: InstanceList, v: int, n_img: int):
        """Per-view flattened probe rows and decoded gt of positives."""
        with torch.no_grad():
            labels_l, bbox_t_l, angle_t_l, bid_l = head.get_targets(
                points, gts)
        flat_labels = torch.cat(labels_l)
        flat_bbox_t = torch.cat(bbox_t_l)
        flat_angle_t = torch.cat(angle_t_l)
        flat_bid = torch.cat(bid_l)
        flat_points = torch.cat([p.repeat(n_img, 1) for p in points])
        flat_rows = torch.cat([
            stash[l][v * n_img:(v + 1) * n_img].permute(0, 2, 3, 1).reshape(
                -1, stash[l].size(1)) for l in range(len(stash))
        ])
        pos = (flat_labels >= 0) & (flat_labels < head.num_classes)
        rows = flat_rows[pos]
        bids = flat_bid[pos]
        if rows.numel() == 0:
            return (flat_rows.new_zeros((0, flat_rows.size(1))),
                    flat_bid.new_zeros((0,)), flat_rows.new_zeros((0, 3)))
        pts = flat_points[pos]
        tgt = torch.cat([flat_bbox_t[pos], flat_angle_t[pos]], dim=-1)
        with torch.no_grad():
            dec = head.bbox_coder.decode(pts, tgt)  # (P, 5): x, y, w, h, th
            w_gt, h_gt, th_gt = dec[:, 2], dec[:, 3], dec[:, 4]
            env_w, env_h = envelope_from_wh_theta(w_gt, h_gt, th_gt)
            extras = torch.stack(
                [env_w, env_h, torch.sin(2.0 * th_gt)], dim=-1)
        return rows, bids, extras

    # -- main hook -------------------------------------------------------

    def _add_auxiliary_losses(self, losses: dict, features: Tuple[Tensor],
                              batch_gt_instances: InstanceList) -> dict:
        head = self.bbox_head
        enabled = self.training and getattr(head, 'gda_enabled', False) \
            and getattr(head, 'loss_gda_probe', None) is not None
        if not enabled:
            return losses
        stash = getattr(head, 'last_gda_probe', None)
        rot_view = getattr(self, '_gda_rot_view', None)
        if not stash or rot_view is None:
            return losses

        rot = rot_view['rot']
        if not isinstance(rot, Tensor):
            rot = stash[0].new_tensor(float(rot))
        gt_ori = batch_gt_instances
        gt_rot = rot_view['gts']
        gt_flp = self._rebuild_flipped_view(gt_ori, gt_rot)
        n_img = len(gt_ori)

        featmap_sizes = [p.shape[-2:] for p in stash]
        points = head.prior_generator.grid_priors(
            featmap_sizes, dtype=stash[0].dtype, device=stash[0].device)

        rows_v, bids_v, extras_v = [], [], []
        for v, gts in enumerate((gt_ori, gt_rot, gt_flp)):
            r, b, e = self._view_tensors(head, stash, points, gts, v, n_img)
            rows_v.append(r)
            bids_v.append(b)
            extras_v.append(e)

        rows3, ext3, _ = compact_probe_by_object(rows_v, bids_v, extras_v)
        loss_dict, diag = head.loss_gda_probe(rows3, ext3[..., :2],
                                              ext3[..., 2], rot)
        losses.update(loss_dict)
        losses.update(diag)
        return losses
