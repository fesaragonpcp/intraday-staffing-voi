"""Sensitivity of the three main conclusions to modeling assumptions.

C1: at the manager's valuations (margin 87.5/600, WTP 150) no signal has decision value.
C2: information has value only within a band of lost-customer value (c_a2 = 250).
C3: arrival counts give at least as much value as abandonment counts.
Outputs sens2.pkl and prints a table.
"""
import numpy as np, pickle, time
import ana
from ana import evaluate, P, J0, xcuts, ncuts
import core3

B = dict(n1=np.array([15., 30., 60.]), n2=np.array([10., 25., 40.]), pm=0.2,
         sr=(10, 15, 20), sw=(30, 40, 50), pat=15.0, shape=1.0)


def mk(**kw):
    p = dict(B); p.update(kw); return p


SCEN = [
    ('Base case', mk()),
    # patience
    ('Patience mean 10 min', mk(pat=10.0)),
    ('Patience mean 21.6 min (15 min = median)', mk(pat=15 / np.log(2))),
    ('Patience Weibull k=0.7 (decreasing hazard)', mk(pdist='weib', shape=0.7)),
    ('Patience Weibull k=2 (increasing hazard)', mk(pdist='weib', shape=2.0)),
    ('Patience lognormal, CV=1', mk(pdist='lognorm', psig=np.sqrt(np.log(2)))),
    # occupancy
    ('Occupancy lognormal, CV=0.5', mk(sdist='lognorm', scv=0.5)),
    ('Occupancy lognormal, CV=1', mk(sdist='lognorm', scv=1.0)),
    ('Occupancy +20% (retail 18, wholesale 48)', mk(sr=(12, 18, 24), sw=(36, 48, 60))),
    ('Wholesale share 10%', mk(pm=0.1)),
    ('Wholesale share 30%', mk(pm=0.3)),
    # demand
    ('Demand overdispersion within state, CV=0.2', mk(odcv=0.2)),
    ('Demand overdispersion within state, CV=0.4', mk(odcv=0.4)),
    ('Flat hourly profile', mk(flat=True)),
    ('Saturated volumes lowered (42.3 / 32.3)', mk(n1=np.array([15., 30., 42.3]), n2=np.array([10., 25., 32.3]))),
    # operation
    ('Transfer delay 15-30 min', mk(delay=(15, 30))),
]

VGRID = list(range(100, 1501, 25))
TOL = 1.0


def summary(d, J=J0, ca2=250.):
    out = {}
    for lab, p in (('margin', P(ca2=ca2)), ('wtp', P(vr=150, vw=150, ca2=ca2)), ('v500', P(vr=500, vw=500, ca2=ca2))):
        r = evaluate(d, p, J)
        out[lab] = dict(static=r['static_pol'], N1=r['V_N1_thr'], X1=r['V_X1_thr'], EVPI=r['V_perfect'],
                        rule=r['N1_thr_pol'], N1free=r['V_N1_free'], X1free=r['V_X1_free'])
    ev = []
    for v in VGRID:
        r = evaluate(d, P(vr=v, vw=v, ca2=ca2), J)
        ev.append((v, r['V_N1_thr'], r['V_X1_thr'], r['V_perfect']))
    ev = np.array(ev)
    pos = ev[ev[:, 1] > TOL, 0]
    out['band'] = (pos.min(), pos.max()) if len(pos) else (np.nan, np.nan)
    inb = ev[ev[:, 1] > TOL]
    out['n1_ge_x1'] = bool(np.all(inb[:, 1] >= inb[:, 2] - TOL)) if len(inb) else True
    out['n1_minus_x1_max'] = float((inb[:, 1] - inb[:, 2]).min()) if len(inb) else np.nan
    out['curve'] = ev
    return out


if __name__ == '__main__':
    res = {}
    for name, p in SCEN:
        t = time.time()
        d = core3.run(p, 4000)
        d['P'] = p
        d['xcut'] = xcuts(d); d['ncut'] = ncuts(p['n1'])
        s = summary(d)
        res[name] = s
        m, w, v = s['margin'], s['wtp'], s['v500']
        print(f"{name:45s} | margin {m['static']} N1 {m['N1']:.1f} X1 {m['X1']:.1f} EVPI {m['EVPI']:.1f} | "
              f"wtp N1 {w['N1']:.1f} X1 {w['X1']:.1f} | v500 {v['static']} N1 {v['N1']:.1f} X1 {v['X1']:.1f} "
              f"EVPI {v['EVPI']:.1f} rule {v['rule']} | band {s['band']} N1>=X1 {s['n1_ge_x1']} "
              f"minDiff {s['n1_minus_x1_max']:.1f} | {time.time()-t:.0f}s", flush=True)
    # prior filling: corners of the unspecified probabilities (base simulations)
    d = ana.S['base']
    priors = {}
    for lab, (ta, sa) in {'even split (base)': (0.5, 0.5), 'typical->calm, saturated->calm': (1, 1),
                          'typical->calm, saturated->typical': (1, 0), 'typical->typical, saturated->calm': (0, 1),
                          'typical->typical, saturated->typical': (0, 0)}.items():
        J = np.array([[0.14, 0.06, 0.0], [0.30 * ta, 0.30 * (1 - ta), 0.0], [0.20 * sa, 0.20 * (1 - sa), 0.30]])
        s = summary(d, J)
        priors[lab] = (J, s)
        m, v = s['margin'], s['v500']
        print(f"PRIOR {lab:40s} | margin N1 {m['N1']:.1f} X1 {m['X1']:.1f} EVPI {m['EVPI']:.1f} | v500 {v['static']} "
              f"N1 {v['N1']:.1f} X1 {v['X1']:.1f} EVPI {v['EVPI']:.1f} rule {v['rule']} | band {s['band']} "
              f"N1>=X1 {s['n1_ge_x1']}", flush=True)
    pickle.dump(dict(scen=res, priors=priors, order=[n for n, _ in SCEN]), open('sens2.pkl', 'wb'))
