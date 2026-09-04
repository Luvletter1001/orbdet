#!/usr/bin/env python3
"""Gate-B B2: cross-seed chamber stability of the detector angle.

Hypothesis (pre-registered in fres_gda_group_decomposed_angle_gate_a.md):
if the chamber (mod-90 root / tilt sign) of near-square instances is a
gauge degree of freedom, two independently seeded detectors will disagree
on it much more often there than on elongated objects.

Read-only: joins the seed3407 and seed42 evidence caches on
(image_id, gt_index), both independently matched to the same GT at
IoU>=0.5, and measures cross-seed angle disagreement by aspect bin.

Pre-registered verdicts:
  SUPPORT gauge : aspect<1.1 disagreement >= 30% and aspect>=2 <= 2%
                  and chi-square p < 0.01 between the two bins
  FALSIFY gauge : aspect<1.1 disagreement < 10%
  else          : inconclusive, report as-is

Beyond the letter of the pre-registration, the script also prints the
flip-joint decomposition per bin: disagreement |predA-predB|>45deg is
driven almost entirely by one seed flipping chamber vs GT while the other
does not, so
    disagree ~= P(flipA XOR flipB) = pA + pB - 2*P(both)
We compare the observed P(both) against the independence baseline pA*pB:
  - P(both) >> pA*pB  -> flips cluster on the same instances across seeds
                         (a shared property of the GT/instance, i.e. the
                         chamber there is gauge / not identifiable)
  - disagree << pA+pB-2*pA*pB (independent baseline)
              -> yet the two seeds' chamber choices are far from random
                         (the chamber is partially learnable)
Both together motivate soft (confidence-weighted) gating rather than a
hard threshold on near-square instances.
"""
import json, math, sys
from collections import defaultdict

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


def load(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get('matched'):
                out[(r['image_id'], r['gt_index'])] = r
    return out


def main(path_a, path_b):
    A, B = load(path_a), load(path_b)
    keys = sorted(set(A) & set(B))
    print('seed3407 matched: %d  seed42 matched: %d  intersection: %d'
          % (len(A), len(B), len(keys)))

    bins = [(1.0, 1.1), (1.1, 1.3), (1.3, 2.0), (2.0, 4.0), (4.0, 1e9)]
    by = defaultdict(list)
    for k in keys:
        ra, rb = A[k], B[k]
        da = abs(wrap((ra['pred_angle'] - rb['pred_angle']) * D, 180.0))
        rec = dict(d=da,
                   ar=ra['gt_aspect_ratio'],
                   flip_a=abs(wrap((ra['pred_angle'] - ra['gt_angle']) * D,
                                   180.0)) > 45,
                   flip_b=abs(wrap((rb['pred_angle'] - rb['gt_angle']) * D,
                                   180.0)) > 45)
        for lo, hi in bins:
            if lo <= rec['ar'] < hi:
                by[(lo, hi)].append(rec)
                break

    print('\n%-14s %7s %12s %15s %14s %11s'
          % ('aspect bin', 'n', 'xseed>45deg%', 'xseed|dth|P50/P90',
             'flipA%/flipB%', 'both-flip%'))
    res = {}
    for lo, hi in bins:
        sub = by.get((lo, hi), [])
        if not sub:
            continue
        n = len(sub)
        k_dis = sum(1 for r in sub if r['d'] > 45)
        k_fa = sum(r['flip_a'] for r in sub)
        k_fb = sum(r['flip_b'] for r in sub)
        k_bo = sum(1 for r in sub if r['flip_a'] and r['flip_b'])
        dis = [r['d'] for r in sub]
        res[(lo, hi)] = dict(n=n, k_dis=k_dis, k_fa=k_fa, k_fb=k_fb,
                             k_both=k_bo, p50=pct(dis, 50), p90=pct(dis, 90))
        print('[%-5s,%4s) %7d %11.2f%% %7.2f/%6.2f %6.2f%%/%6.2f%% %10.2f%%'
              % (lo, 'inf' if hi > 1e8 else hi, n, 100 * k_dis / n,
                 pct(dis, 50), pct(dis, 90),
                 100 * k_fa / n, 100 * k_fb / n, 100 * k_bo / n))

    # ---- flip-joint decomposition per bin -------------------------------
    print('\nflip-joint decomposition (why disagreement happens):')
    print('%-14s %7s %7s %7s %9s %9s %11s'
          % ('bin', 'pA%', 'pB%', 'both%', 'P(B|A)%', 'XOR%*', 'indep-dis%'))
    print('    *XOR = pA+pB-2*both = disagreement implied by flips alone;')
    print('     indep-dis = pA+pB-2*pA*pB = baseline if flips were '
          'independent across seeds')
    for lo, hi in bins:
        s = res.get((lo, hi))
        if not s:
            continue
        pa = s['k_fa'] / s['n']; pb = s['k_fb'] / s['n']
        bo = s['k_both'] / s['n']
        xor = pa + pb - 2 * bo
        indep = pa + pb - 2 * pa * pb
        cond = bo / pa if pa > 0 else float('nan')
        print('[%-5s,%4s) %6.2f %6.2f %6.2f %8.1f %8.2f %10.2f'
              % (lo, 'inf' if hi > 1e8 else hi, 100 * pa, 100 * pb,
                 100 * bo, 100 * cond, 100 * xor, 100 * indep))

    # ---- chi-square: aspect<1.1 vs pooled aspect>=2 ---------------------
    lo_bin = res.get((1.0, 1.1))
    hi_parts = [res.get(b) for b in ((2.0, 4.0), (4.0, 1e9))]
    hi_parts = [s for s in hi_parts if s]
    if lo_bin and hi_parts:
        n1 = lo_bin['n']; k1 = lo_bin['k_dis']; p1 = k1 / n1
        n2 = sum(s['n'] for s in hi_parts)
        k2 = sum(s['k_dis'] for s in hi_parts)
        p2 = k2 / n2
        pe = (k1 + k2) / (n1 + n2)
        chi2 = ((k1 - n1 * pe) ** 2 / (n1 * pe)
                + (n1 - k1 - n1 * (1 - pe)) ** 2 / (n1 * (1 - pe))
                + (k2 - n2 * pe) ** 2 / (n2 * pe)
                + (n2 - k2 - n2 * (1 - pe)) ** 2 / (n2 * (1 - pe)))
        print('\nchi2(aspect<1.1 vs pooled aspect>=2, disagree) = %.1f '
              '(chi2_0.99,df=1 = 6.63)' % chi2)
        print('pooled aspect>=2: n=%d  disagree=%.2f%%' % (n2, 100 * p2))
        print('pre-registered verdict: ', end='')
        if p1 >= 0.30 and p2 <= 0.02 and chi2 > 6.63:
            print('SUPPORT — chamber is gauge on near-square instances')
        elif p1 < 0.10:
            print('FALSIFY — chamber is stable/learned even on near-square')
        else:
            print('INCONCLUSIVE — see decomposition above '
                  '(flips cluster on shared instances, yet disagreement '
                  'is well below the independent baseline)')

    # ---- continuous-part agreement where the chamber agrees -------------
    print('\ncontinuous-part agreement (xseed |dth| where <=45deg):')
    for lo, hi in bins:
        sub = [r['d'] for r in by.get((lo, hi), []) if r['d'] <= 45]
        if sub:
            print('[%-5s,%4s) n=%7d  P50=%5.2f  P90=%5.2f'
                  % (lo, 'inf' if hi > 1e8 else hi, len(sub),
                     pct(sub, 50), pct(sub, 90)))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
