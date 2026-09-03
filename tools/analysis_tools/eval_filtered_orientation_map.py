#!/usr/bin/env python3
"""End-to-end mAP evaluation of geometric-suspicion filtering.

Runs the frozen checkpoint over the grouped holdout, computes train-free
geometric cues (image-arbitrated low-rank axis + confidence) for every
*prediction*, ranks a fused suspicion score, and measures rotated mAP
(DOTAMetric, IoU=0.5, 11-point) for:

  - baseline          : untouched predictions
  - remove@p          : drop the top-p%% most suspicious predictions
  - downweight@p      : multiply flagged predictions' scores by a factor

Read-only: never trains, never writes source data; predictions and cues
are recomputed in memory. Mirrors the collector's frozen-inference guards.
"""

import argparse
import copy
import json
import math
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.structures import InstanceData

from mmdet.structures.bbox import bbox2roi
from mmrotate.models.losses.low_rank_orientation_evidence import (
    low_rank_channel_orientation_evidence)
from mmrotate.structures import RotatedBoxes
from mmrotate.utils import register_all_modules

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.analysis_tools.collect_low_rank_orientation_evidence import (  # noqa: E402
    _box_tensor, _square_hboxes, _validate_grouped_holdout,
    build_roi_extractor, image_patch_axis, sha256_file)

D = 180.0 / math.pi
# orientation-sensitive elongated classes (DOTA label ids) for per-class AP
FOCUS_CLASSES = {'plane': 0, 'small-vehicle': 4, 'large-vehicle': 5,
                 'ship': 6, 'helicopter': 14}


def wrap_deg(x, period=180.0):
    return (x + period / 2) % period - period / 2


def ranks01(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    for pos, i in enumerate(order):
        out[i] = pos / max(len(values) - 1, 1)
    return out


def fuse_suspicion(disagree_deg: Sequence[float],
                   one_minus_conf: Sequence[float]) -> List[float]:
    """Rank-sum fusion of geometric disagreement and low confidence."""
    rd = ranks01(disagree_deg)
    rc = ranks01(one_minus_conf)
    return [a + b for a, b in zip(rd, rc)]


def apply_filter(bboxes: torch.Tensor, scores: torch.Tensor,
                 labels: torch.Tensor, flagged: torch.Tensor, mode: str,
                 downweight: float):
    """Return (bboxes, scores, labels) after removing or downweighting the
    flagged predictions. Pure function, unit-testable."""
    if mode == 'remove':
        keep = ~flagged
        return bboxes[keep], scores[keep], labels[keep]
    if mode == 'downweight':
        new_scores = scores.clone()
        new_scores[flagged] = new_scores[flagged] * downweight
        return bboxes, new_scores, labels
    raise ValueError(f'unknown filter mode: {mode}')


def prediction_cues(pred_instances, scale_factor, features, image_index,
                    inputs, roi_extractor):
    """Geometric suspicion components for one image's predictions.

    Returns (disagree_deg[list], one_minus_conf[list]) aligned with
    pred_instances order. Empty predictions -> empty lists.
    """
    n = pred_instances.bboxes.shape[0]
    if n == 0:
        return [], []
    rbox = _box_tensor(pred_instances.bboxes).detach().to(torch.float32)
    sw, sh = float(scale_factor[0]), float(scale_factor[1])
    scale = rbox.new_tensor([sw, sh, sw, sh, 1.0])
    feat_rbox = RotatedBoxes(rbox * scale)
    hbox = feat_rbox.convert_to('hbox').tensor
    square = _square_hboxes(hbox)
    rois = bbox2roi([square]).to(device=features[0].device,
                                 dtype=features[0].dtype)
    roi_features = roi_extractor(
        tuple(feat.detach() for feat in features), rois)
    evidence = low_rank_channel_orientation_evidence(
        roi_features, min_energy=1e-6, min_anisotropy=0.1, eps=1e-6)
    image_axis = image_patch_axis(inputs[image_index:image_index + 1],
                                  [square])
    pred_angle = torch.rad2deg(rbox[:, 4])
    lr = torch.rad2deg(evidence['axis_angle'].cpu())
    im = torch.rad2deg(image_axis.cpu())
    combo = torch.stack([lr, lr + 90.0], dim=0)
    pick = (wrap_deg(combo - im.unsqueeze(0)).abs().argmin(dim=0))
    axis = combo[pick, torch.arange(n)]
    disagree = wrap_deg(pred_angle.cpu() - axis).abs()
    one_minus_conf = 1.0 - evidence['confidence'].detach().cpu()
    return disagree.tolist(), one_minus_conf.tolist()


def evaluate_variant(cfg, dataset_meta, template_samples, kept_preds,
                     out_prefix=None):
    """Build modified data samples and run DOTAMetric offline."""
    from mmengine.evaluator import Evaluator
    evaluator = Evaluator([dict(cfg.val_evaluator)])
    evaluator.dataset_meta = dataset_meta
    modified = []
    for sample, pred in zip(template_samples, kept_preds):
        new_sample = copy.deepcopy(sample)
        new_pred = InstanceData()
        new_pred.bboxes = RotatedBoxes(pred['bboxes'])
        new_pred.scores = pred['scores']
        new_pred.labels = pred['labels']
        new_sample.pred_instances = new_pred
        modified.append(new_sample)
    evaluator.process(modified)
    return evaluator.evaluate(len(modified))


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='metrics JSON path (refuses to overwrite)')
    parser.add_argument('--max-images', type=int)
    parser.add_argument('--rates', type=float, nargs='+',
                        default=[0.05, 0.10])
    parser.add_argument('--downweight', type=float, default=0.3)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f'output exists: {output_path}')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config_digest = sha256_file(config_path)
    checkpoint_digest = sha256_file(checkpoint_path)

    register_all_modules()
    cfg = Config.fromfile(config_path)
    _validate_grouped_holdout(cfg)
    cfg.launcher = 'none'
    cfg.load_from = None
    cfg.resume = False
    with tempfile.TemporaryDirectory(prefix='orbdet_map_eval_') as ws:
        cfg.work_dir = ws
        runner = Runner.from_cfg(cfg)
        runner.load_checkpoint(str(checkpoint_path))
        if (sha256_file(config_path) != config_digest
                or sha256_file(checkpoint_path) != checkpoint_digest):
            raise RuntimeError('config or checkpoint changed while loading')
        model = runner.model.eval()
        device = next(model.parameters()).device
        roi_extractor = build_roi_extractor().to(device).eval()
        dataset_meta = runner.val_dataloader.dataset.metainfo

        samples = []          # data samples carrying GT (+ original preds)
        pred_store = []       # per image: dict(bboxes, scores, labels)
        disagree_all: List[float] = []
        conf_all: List[float] = []
        image_count = 0
        with torch.inference_mode():
            for data in runner.val_dataloader:
                if args.max_images is not None and image_count >= (
                        args.max_images):
                    break
                processed = model.data_preprocessor(data, training=False)
                inputs = processed['inputs']
                batch_samples = list(processed['data_samples'])
                features = model.extract_feat(inputs)
                preds = list(model.predict(inputs, batch_samples,
                                           rescale=True))
                for i, (sample, pred) in enumerate(zip(batch_samples,
                                                       preds)):
                    if args.max_images is not None and (
                            image_count >= args.max_images):
                        break
                    pred_instances = pred.pred_instances
                    disagree, one_minus_conf = prediction_cues(
                        pred_instances,
                        sample.metainfo['scale_factor'],
                        features, i, inputs, roi_extractor)
                    disagree_all.extend(disagree)
                    conf_all.extend(one_minus_conf)
                    pred_store.append(dict(
                        bboxes=_box_tensor(pred_instances.bboxes).detach(
                        ).cpu(),
                        scores=pred_instances.scores.detach().cpu(),
                        labels=pred_instances.labels.detach().cpu()))
                    samples.append(pred)  # carries gt_instances + img_id
                    image_count += 1
                    if image_count % 200 == 0:
                        print(f'[eval] {image_count} images', flush=True)

    suspicion = fuse_suspicion(disagree_all, conf_all)
    order = sorted(range(len(suspicion)), key=lambda i: -suspicion[i])
    rank_of = [0] * len(suspicion)
    for pos, i in enumerate(order):
        rank_of[i] = pos
    total_pred = len(suspicion)

    def variant_preds(rate, mode):
        offset = 0
        out = []
        for pred in pred_store:
            n = pred['bboxes'].shape[0]
            if n:
                idx = torch.arange(offset, offset + n)
                flagged = torch.tensor(
                    [rank_of[j] < rate * total_pred for j in idx.tolist()])
                b, s, l = apply_filter(pred['bboxes'], pred['scores'],
                                       pred['labels'], flagged, mode,
                                       args.downweight)
                out.append(dict(bboxes=b, scores=s, labels=l))
            else:
                out.append(pred)
            offset += n
        return out

    results = {}
    print('[eval] baseline mAP ...', flush=True)
    results['baseline'] = evaluate_variant(
        cfg, dataset_meta, samples, pred_store)
    print('  ->', {k: round(v, 4) for k, v in results['baseline'].items()},
          flush=True)
    for rate in args.rates:
        for mode in ('remove', 'downweight'):
            name = f'{mode}@{rate:g}'
            print(f'[eval] {name} ...', flush=True)
            results[name] = evaluate_variant(
                cfg, dataset_meta, samples, variant_preds(rate, mode))
            print('  ->', {k: round(v, 4) for k, v in results[name].items()},
                  flush=True)

    # per-class AP for orientation-sensitive classes, baseline vs remove@5%
    per_class = {}
    for cls_name, cls_id in FOCUS_CLASSES.items():
        for variant, rate, mode in (('baseline', 0.0, 'remove'),
                                    ('remove@0.05', 0.05, 'remove')):
            subset_samples, subset_preds = [], []
            offset = 0
            for sample, pred in zip(samples, pred_store):
                n = pred['bboxes'].shape[0]
                flagged = torch.zeros(n, dtype=torch.bool)
                if n and rate:
                    idx = range(offset, offset + n)
                    flagged = torch.tensor(
                        [rank_of[j] < rate * total_pred for j in idx])
                keep_lbl = pred['labels'] == cls_id
                b, s, l = apply_filter(pred['bboxes'], pred['scores'],
                                       pred['labels'], flagged, mode,
                                       args.downweight)
                keep_lbl = l == cls_id
                subset_preds.append(dict(bboxes=b[keep_lbl],
                                         scores=s[keep_lbl],
                                         labels=l[keep_lbl]))
                s2 = copy.deepcopy(sample)
                gt = s2.gt_instances
                mask = gt.labels == cls_id
                gt2 = InstanceData()
                gt2.bboxes = gt.bboxes[mask]
                gt2.labels = gt.labels[mask]
                s2.gt_instances = gt2
                subset_samples.append(s2)
                offset += n
            per_class[f'{cls_name}:{variant}'] = evaluate_variant(
                cfg, dataset_meta, subset_samples, subset_preds)
            print(f'[eval] class {cls_name} {variant} ->',
                  {k: round(v, 4)
                   for k, v in per_class[f'{cls_name}:{variant}'].items()},
                  flush=True)

    payload = dict(
        config=str(config_path), config_sha256=config_digest,
        checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_digest,
        image_count=image_count, prediction_count=total_pred,
        rates=args.rates, downweight=args.downweight,
        overall=results, per_class=per_class)
    with output_path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2,
                  sort_keys=True)
    print('saved:', output_path)


if __name__ == '__main__':
    main()
