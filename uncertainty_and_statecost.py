import numpy as np, pickle
from ana import *
# --- state-dependent opportunity cost
def TCs(d,p,ca1s,ca2s):
    i=np.array([0,1])[None,None,None,:,None]; j=np.array([0,1])[None,None,None,None,:]
    c1=np.array(ca1s)[None,:,None,None,None]; c2=np.array(ca2s)[None,None,:,None,None]
    return c1*i*8+c2*j*6+p['co']*(2+j)*d['H']+p['vr']*d['Lr']+p['vw']*d['Lw']
import ana
orig=ana.TC
res={}
for lab,p in (('margin',P()),('v500',P(vr=500,vw=500))):
    ana.TC=lambda d,pp: TCs(d,pp,(80,120,200),(150,250,400))
    r=ana.evaluate(S['base'],p,J0); ana.TC=orig
    res[lab]=(r['static_pol'],r['N1_thr_pol'],round(r['V_N1_thr'],1),round(r['V_X1_thr'],1),round(r['V_perfect'],1))
    print('state-dependent c_a',lab,res[lab])
pickle.dump(res,open('statecost.pkl','wb'))
# --- uncertainty rerun storing raw
keys=['base','pat10','pat20','weib2','may10','may30']; pos=J0>0
def fixed_rule_value(d,p,J,i,t):
    tc=TC(d,p); tr=tc[TR]; te=tc[TE]
    st={(a,b):wmean(J,tr[...,a,b]) for a in (0,1) for b in (0,1)}; a0,b0=min(st,key=st.get)
    s=d['N1'][TE]; return wmean(J,te[...,a0,b0])-wmean(J,te[...,i,0]+(te[...,i,1]-te[...,i,0])*(s[:,:,None]>=t))
raw={}
for lab,vv in (('margin',None),('v500',500.)):
    for conc in (20,50,200):
        rng=np.random.default_rng(2026); out=[]
        for k in range(200):
            sk=keys[rng.integers(6)]; f=rng.uniform(0.8,1.2)
            vr,vw=(87.5*f,600*f) if vv is None else (vv*f,vv*f)
            p=dict(ca1=rng.triangular(80,120,200),ca2=rng.triangular(150,250,400),co=85.,vr=vr,vw=vw)
            J=np.zeros((3,3)); J[pos]=rng.dirichlet(J0[pos]*conc)
            r=evaluate(S[sk],p,J); fx=fixed_rule_value(S[sk],p,J,1,43) if vv else np.nan
            out.append((r['V_N1_thr'],r['V_X1_thr'],r['V_perfect'],fx))
        raw[(lab,conc)]=np.array(out)
        O=raw[(lab,conc)]
        print(lab,conc,'P5/50/95 N1',np.percentile(O[:,0],[5,50,95]).round(1),'P(>0)',(O[:,0]>0).mean(),'P(>1)',(O[:,0]>1).mean(),'P(N1>=X1-1)',(O[:,0]>=O[:,1]-1).mean(), 'P(N1>X1)',(O[:,0]>O[:,1]).mean(),
              'EVPI P50',np.percentile(O[:,2],50).round(1), 'fixed>0',np.nanmean(O[:,3]>0) if vv else '', 'fixed P5/50/95',np.nanpercentile(O[:,3],[5,50,95]).round(1) if vv else '',flush=True)
pickle.dump(raw,open('unc_raw.pkl','wb'))
