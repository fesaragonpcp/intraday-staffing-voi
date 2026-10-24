"""Extended simulation for the sensitivity analysis of modeling assumptions.

Same event logic (core2.day) and seed structure as core2.run; the customer
stream adds options for:
  patience distribution  P['pdist'] in {'exp','weib','lognorm'} with P['shape'] or P['psig']
  occupancy distribution P['sdist'] in {'tri','lognorm'} with P['scv'] (coefficient of variation)
  demand overdispersion  P['odcv'] > 0: block volume multiplied by a Gamma(1/cv^2, cv^2) factor
  hourly profile         P['flat'] = True uses a flat profile within each block
  transfer delay         P['delay'] = (low, high) minutes, uniform
"""
import numpy as np
from math import gamma, log, sqrt, exp
from core2 import day, T1, wM, wA

FLAT_M = np.ones(8) / 8
FLAT_A = np.ones(6) / 6


def _lognorm(rng, mean, cv, m):
    s2 = log(1 + cv * cv)
    return rng.lognormal(log(mean) - s2 / 2, sqrt(s2), m)


def stream(rng, n, w, off, P):
    if P.get('odcv', 0) > 0:
        k_ = 1 / P['odcv'] ** 2
        n = n * rng.gamma(k_, 1 / k_)
    k = rng.poisson(n * w)
    arr = np.sort(np.concatenate([off + h * 60 + rng.uniform(0, 60, x) for h, x in enumerate(k)]))
    m = len(arr)
    may = rng.random(m) < P['pm']
    if P.get('sdist', 'tri') == 'tri':
        sv = np.where(may, rng.triangular(*P['sw'], m), rng.triangular(*P['sr'], m))
    else:
        sv = np.where(may, _lognorm(rng, P['sw'][1], P['scv'], m), _lognorm(rng, P['sr'][1], P['scv'], m))
    pd = P.get('pdist', 'exp')
    if pd == 'exp':
        pat = rng.exponential(P['pat'], m)
    elif pd == 'weib':
        pat = P['pat'] / gamma(1 + 1 / P['shape']) * rng.weibull(P['shape'], m)
    else:  # lognormal with given sigma, mean = P['pat']
        s = P['psig']
        pat = rng.lognormal(log(P['pat']) - s * s / 2, s, m)
    return arr, sv, pat, may


def run(P, R, seed0=0):
    n1, n2 = P['n1'], P['n2']
    wm = FLAT_M if P.get('flat') else wM
    wa = FLAT_A if P.get('flat') else wA
    lo, hi = P.get('delay', (5, 10))
    sh = (R, 3, 3, 2, 2)
    out = {k: np.zeros(sh, np.int32) for k in ('X1', 'Lr', 'Lw', 'L1', 'L2', 'Q16', 'S')}
    out['H'] = np.zeros(sh)
    out['N1'] = np.zeros((R, 3), np.int32)
    out['N2'] = np.zeros((R, 3), np.int32)
    for r in range(R):
        ms = [stream(np.random.default_rng([seed0, r, 1, a]), n1[a], wm, 0.0, P) for a in range(3)]
        dl = np.random.default_rng([seed0, r, 9]).uniform(lo, hi)
        af = [stream(np.random.default_rng([seed0, r, 2, b]), n2[b], wa, T1, P) for b in range(3)]
        for a in range(3):
            out['N1'][r, a] = len(ms[a][0])
            for b in range(3):
                if a == 0:
                    out['N2'][r, b] = len(af[b][0])
                arr = np.concatenate([ms[a][0], af[b][0]])
                sv = np.concatenate([ms[a][1], af[b][1]])
                pt = np.concatenate([ms[a][2], af[b][2]])
                my = np.concatenate([ms[a][3], af[b][3]])
                for i, d1 in enumerate((2, 3)):
                    for j, d2 in enumerate((2, 3)):
                        x, lr, lw, h, l1, l2, q, s = day(arr, sv, pt, my, d1, d2, dl)
                        for kk, v in (('X1', x), ('Lr', lr), ('Lw', lw), ('H', h), ('L1', l1),
                                      ('L2', l2), ('Q16', q), ('S', s)):
                            out[kk][r, a, b, i, j] = v
    return out
