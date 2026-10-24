"""Supporting estimates used by the decision model (base case).

T-S1  distribution of the signals N1 and X1 by morning state (and morning staffing for X1)
T-S2  simulated consequences by state pair and staffing strategy, with Monte Carlo standard errors
T-S3  derivation of the joint prior from the elicited statements
Outputs support.pkl and prints the tables.
"""
import numpy as np, pickle
from scipy import stats
from ana import S, J0

d = S['base']
P = d['P']
R = d['N1'].shape[0]
st = ['calm', 'typical', 'saturated']
out = {}

# T-S1 signals
sig = []
for a in range(3):
    n = d['N1'][:, a]
    sig.append(('N1', st[a], '-', n.mean(), n.std(ddof=1), *np.percentile(n, [5, 50, 95]), P['n1'][a]))
for i, c in ((0, 2), (1, 3)):
    for a in range(3):
        x = d['X1'][:, a, 0, i, 0]
        sig.append(('X1', st[a], c, x.mean(), x.std(ddof=1), *np.percentile(x, [5, 50, 95]), np.nan))
print('T-S1 signal distributions (4000 replications per state)')
for s in sig:
    print(f"{s[0]} {s[1]:9s} d1={s[2]} mean {s[3]:.2f} sd {s[4]:.2f} P5/50/95 {s[5]:.0f}/{s[6]:.0f}/{s[7]:.0f}")
out['signals'] = sig

# overlap between adjacent states (total-variation distance) for each signal
tv = []
for name, arrs in (('N1', [d['N1'][:, a] for a in range(3)]),
                   ('X1 d1=2', [d['X1'][:, a, 0, 0, 0] for a in range(3)]),
                   ('X1 d1=3', [d['X1'][:, a, 0, 1, 0] for a in range(3)])):
    M = max(x.max() for x in arrs) + 1
    p = [np.bincount(x, minlength=M) / len(x) for x in arrs]
    tv.append((name, 0.5 * np.abs(p[0] - p[1]).sum(), 0.5 * np.abs(p[1] - p[2]).sum(), 0.5 * np.abs(p[0] - p[2]).sum()))
print('Total-variation distance between state-conditional distributions (calm-typ, typ-sat, calm-sat)')
for t in tv:
    print(f"{t[0]:8s} {t[1]:.3f} {t[2]:.3f} {t[3]:.3f}")
out['tv'] = tv

# T-S2 consequences
L = (d['Lr'] + d['Lw']).astype(float)
H = d['H']
cons = []
for a in range(3):
    for b in range(3):
        row = [st[a], st[b], J0[a, b]]
        for i in (0, 1):
            for j in (0, 1):
                l = L[:, a, b, i, j]; h = H[:, a, b, i, j]
                row += [l.mean(), l.std(ddof=1) / np.sqrt(R), h.mean() * 60, h.std(ddof=1) * 60 / np.sqrt(R),
                        d['Lw'][:, a, b, i, j].mean()]
        cons.append(row)
print('T-S2 mean lost customers (SE) and overtime minutes by state pair; strategies (2,2),(2,3),(3,2),(3,3)')
for r in cons:
    s = f"{r[0]:9s} {r[1]:9s} p={r[2]:.2f} |"
    for k in range(4):
        b = 3 + 5 * k
        s += f" L {r[b]:.2f} ({r[b+1]:.2f}) H {r[b+2]:.1f} |"
    print(s)
out['cons'] = cons

# T-S3 prior derivation
pm = np.array([0.2, 0.3, 0.5]); ps = 0.3
deriv = dict(
    morning_marginal=pm,
    sat_sat=0.5 * 0.6, calm_calm=0.2 * 0.7, calm_typ=0.2 * 0.3,
    sat_afternoons_from_others=ps - 0.5 * 0.6,
    J=J0, afternoon_marginal=J0.sum(0))
print('T-S3', deriv)
out['prior'] = deriv
pickle.dump(out, open('support.pkl', 'wb'))
