import numpy as np, pickle
from ana import *
d=S['base']; N1=d['N1']
def evsi_n1(p,J=J0):
    tc=TC(d,p); tr=tc[TR]; te=tc[TE]
    st={(i,j):wmean(J,tr[...,i,j]) for i in (0,1) for j in (0,1)}; i0,j0=min(st,key=st.get)
    base_te=wmean(J,te[...,i0,j0]); best=(1e18,)
    for i in (0,1):
        s=N1[TR]; diff=tr[...,i,1]-tr[...,i,0]; b0=wmean(J,tr[...,i,0])
        for t in range(0,N1.max()+2):
            c=b0+wmean(J,diff*(s[:,:,None]>=t))
            if c<best[0]: best=(c,i,t)
    _,i,t=best; s=N1[TE]
    return base_te-wmean(J,te[...,i,0]+(te[...,i,1]-te[...,i,0])*(s[:,:,None]>=t))
V=list(range(100,1001,50)); C=list(range(80,401,40))
G=np.array([[max(0,evsi_n1(P(vr=v,vw=v,ca2=float(c)))) for v in V] for c in C])
pickle.dump((V,C,G),open('grid.pkl','wb')); print(G.round(0))
