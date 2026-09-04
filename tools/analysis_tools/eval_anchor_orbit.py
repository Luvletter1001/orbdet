#!/usr/bin/env python3
"""Anchor+orbit protocol evaluation on the cached orientation evidence.

Protocol (read-only, post-hoc on detector outputs):
  anchor  = detector angle (e2). Its continuous part is never re-estimated.
  orbit   = discrete root k in {0,1} (90-deg). External geometric evidence
            may vote on the root, or fully override, ONLY on instances that
            are (a) flagged by the fused QC signal and (b) have
            self-consistent external evidence.

Variants on flag & confident:
  V1 root-vote : theta = pred + 90*k*, k* = argmin_k |wrap180(pred+90k-ext)|
  V2 override  : theta = ext (image-arbitrated low-rank axis)
  V3 hybrid    : disagree>45 -> root-vote ; disagree>15 -> override ; else keep

Two scoring metrics:
  e2 口径      : |wrap180(theta - gt)|
  双根命中口径 : min over the two roots = |wrap90(theta - gt)|  (e4)
"""
import json, math
from collections import defaultdict

PATH = '/Users/zcy/Documents/kimi/tasks/2026-09-03/09-00-59-f8e7bc8b/evidence_cues.jsonl'
D = 180.0 / math.pi


def wrap(x, p):
    return (x + p / 2) % p - p / 2


def pct(v, p):
    s = sorted(v)
    if not s:
        return float('nan')
    k = (len(s) - 1) * p / 100.0
    f = int(k); c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def ranks01(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    for pos, i in enumerate(order):
        out[i] = pos / max(len(values) - 1, 1)
    return out


rows = []
with open(PATH) as fh:
    for line in fh:
        r = json.loads(line)
        if not (r.get('matched') and r.get('low_rank_valid')):
            continue
        gt = r['gt_angle'] * D
        pred = r['pred_angle'] * D
        lr = r['low_rank_angle'] * D
        im = r['image_axis'] * D
        # image-arbitrated low-rank axis (our best geometric estimator)
        ext = min((lr, lr + 90.0), key=lambda a: abs(wrap(a - im, 180.0)))
        r['_gt'] = gt; r['_pred'] = pred; r['_ext'] = ext
        r['_agree'] = abs(wrap(lr - im, 180.0))          # evidence self-consistency
        r['_disagree'] = abs(wrap(pred - ext, 180.0))    # detector vs evidence
        rows.append(r)

n = len(rows)
print('valid+matched:', n)

# fused QC suspicion: rank(disagreement) + rank(1 - low_rank_confidence)
# (reproduces the reported AUROC 0.819 / top5% capture 30.1%; detector's own
#  pred_score is useless here, AUROC 0.487 alone)
qc = [a + b for a, b in zip(ranks01([r['_disagree'] for r in rows]),
                            ranks01([1.0 - r['low_rank_confidence']
                                     for r in rows]))]
for r, q in zip(rows, qc):
    r['_qc'] = q

# quick AUROC sanity of qc for e2_deg > 15 (should be ~0.82 as before)
pos = [r['_qc'] for r in rows if r['e2_deg'] > 15]
neg = [r['_qc'] for r in rows if r['e2_deg'] <= 15]
tot = len(pos) * len(neg)
w = sum(1 for a in pos for b in neg if a > b) + 0.5 * sum(
    1 for a in pos for b in neg if a == b)
print('QC AUROC (e2>15deg): %.3f  (pos=%d)' % (w / tot, len(pos)))


def outcome(r, variant):
    """Return final angle (deg) under the given protocol variant."""
    act = r['_flag'] and r['_conf']
    if not act:
        return r['_pred']
    if variant == 'V1':
        return min((r['_pred'], r['_pred'] + 90.0),
                   key=lambda a: abs(wrap(a - r['_ext'], 180.0)))
    if variant == 'V2':
        return r['_ext']
    if variant == 'V3':
        if r['_disagree'] > 45.0:
            return min((r['_pred'], r['_pred'] + 90.0),
                       key=lambda a: abs(wrap(a - r['_ext'], 180.0)))
        if r['_disagree'] > 15.0:
            return r['_ext']
        return r['_pred']
    raise ValueError(variant)


def report(tag, errs_e2, errs_e4, acted_mask):
    acted = [i for i, m in enumerate(acted_mask) if m]
    line = ('%-26s P50=%6.2f P90=%6.2f >10°=%4.1f%% >15°=%4.1f%% flip>60=%4.1f%%'
            % (tag, pct(errs_e2, 50), pct(errs_e2, 90),
               100 * sum(1 for e in errs_e2 if e > 10) / len(errs_e2),
               100 * sum(1 for e in errs_e2 if e > 15) / len(errs_e2),
               100 * sum(1 for e in errs_e2 if e > 60) / len(errs_e2)))
    if acted:
        sub2 = [errs_e2[i] for i in acted]
        sub4 = [errs_e4[i] for i in acted]
        line += (' | acted n=%d: e2 P50=%.2f P90=%.2f, e4 P50=%.2f P90=%.2f'
                 % (len(acted), pct(sub2, 50), pct(sub2, 90),
                    pct(sub4, 50), pct(sub4, 90)))
    print(line)


for conf_thr in (15.0,):
    for flag_pct in (5.0, 10.0, 20.0):
        thr = pct([r['_qc'] for r in rows], 100.0 - flag_pct)
        for r in rows:
            r['_flag'] = r['_qc'] >= thr
            r['_conf'] = r['_agree'] <= conf_thr
        acted = [r['_flag'] and r['_conf'] for r in rows]
        print('\n=== flag top %g%% QC, confident: |lr-im|<=%g° -> acted %d (%.1f%%) ==='
              % (flag_pct, conf_thr, sum(acted), 100 * sum(acted) / n))

        base_e2 = [r['e2_deg'] for r in rows]
        base_e4 = [r['e4_deg'] for r in rows]
        report('detector e2 (baseline)', base_e2, base_e4, acted)

        for variant in ('V1', 'V2', 'V3'):
            outs = [outcome(r, variant) for r in rows]
            e2s = [abs(wrap(o - r['_gt'], 180.0)) for o, r in zip(outs, rows)]
            e4s = [abs(wrap(o - r['_gt'], 90.0)) for o, r in zip(outs, rows)]
            report('%s anchor+orbit' % variant, e2s, e4s, acted)

        # oracle upper bound: override iff it truly helps (cheating, headroom)
        outs = []
        for r in rows:
            if r['_flag'] and r['_conf']:
                keep = abs(wrap(r['_pred'] - r['_gt'], 180.0))
                over = abs(wrap(r['_ext'] - r['_gt'], 180.0))
                outs.append(r['_ext'] if over < keep else r['_pred'])
            else:
                outs.append(r['_pred'])
        e2s = [abs(wrap(o - r['_gt'], 180.0)) for o, r in zip(outs, rows)]
        e4s = [abs(wrap(o - r['_gt'], 90.0)) for o, r in zip(outs, rows)]
        report('oracle-override (headroom)', e2s, e4s, acted)

# standalone estimators under the dual-root (best-of-two-peaks) metric
print('\n=== standalone estimators, 双根命中口径 |wrap90| ===')
ests = {
    'low-rank axis': [r['low_rank_angle'] * D for r in rows],
    'image axis': [r['image_axis'] * D for r in rows],
    'lr+image arbitration': [r['_ext'] for r in rows],
    'detector (stored e4)': None,
}
print('%-26s %8s %8s %9s %9s' % ('estimator', 'P50', 'P90', '<=10°%', '<=15°%'))
for name, vals in ests.items():
    if vals is None:
        errs = [r['e4_deg'] for r in rows]
    else:
        errs = [abs(wrap(v - r['_gt'], 90.0)) for v, r in zip(vals, rows)]
    print('%-26s %8.2f %8.2f %8.1f%% %8.1f%%'
          % (name, pct(errs, 50), pct(errs, 90),
             100 * sum(1 for e in errs if e <= 10) / n,
             100 * sum(1 for e in errs if e <= 15) / n))

# acted-subset class breakdown for V3 at 10% flag
print('\n=== V3 @10%% flag: acted subset by aspect bin ===')
thr = pct([r['_qc'] for r in rows], 90.0)
for r in rows:
    r['_flag'] = r['_qc'] >= thr
    r['_conf'] = r['_agree'] <= 15.0
for lo, hi in [(1.0, 1.3), (1.3, 2.0), (2.0, 4.0), (4.0, 1e9)]:
    sub = [r for r in rows if lo <= r['gt_aspect_ratio'] < hi
           and r['_flag'] and r['_conf']]
    if len(sub) < 20:
        continue
    b = [r['e2_deg'] for r in sub]
    o = [abs(wrap(outcome(r, 'V3') - r['_gt'], 180.0)) for r in sub]
    print('[%-4g,%4s) acted n=%5d | det e2 P50=%6.2f P90=%6.2f -> V3 P50=%6.2f P90=%6.2f'
          % (lo, 'inf' if hi > 1e8 else hi, len(sub),
             pct(b, 50), pct(b, 90), pct(o, 50), pct(o, 90)))
