import numpy as np, numba as nb
from math import gamma
T1=480.0; TC=840.0
wM=np.array([0.5,0.8,1.1,1.5,1.2,1.0,0.8,0.7]); wM=wM/wM.sum()
wA=np.array([1.0,1.4,1.2,1.0,0.8,0.5]); wA=wA/wA.sum()
@nb.njit(cache=True)
def day(arr,svc,pat,may,d1,d2,delay):
    free=np.zeros(3); frm=np.zeros(3); unt=np.full(3,1e18)
    if d1==2 and d2==2: frm[2]=1e18
    elif d1==2 and d2==3: frm[2]=T1+delay
    elif d1==3 and d2==2: unt[2]=T1
    X1=0; Lr=0; Lw=0; L1=0; L2=0; Q16=0; last=0.0; served=0; N=arr.shape[0]
    for i in range(N):
        a=arr[i]; best=1e18; bj=-1
        for j in range(3):
            s=max(a,free[j],frm[j])
            if s<unt[j] and s<best: best=s; bj=j
        if best-a>pat[i]:
            if a+pat[i]<T1: X1+=1
            if a<T1: L1+=1
            else: L2+=1
            if may[i]: Lw+=1
            else: Lr+=1
            if a<T1 and a+pat[i]>T1: Q16+=1
        else:
            served+=1
            if a<T1 and best>T1: Q16+=1
            free[bj]=best+svc[i]
            if free[bj]>last: last=free[bj]
    H=max(0.0,last-TC)/60.0
    return X1,Lr,Lw,H,L1,L2,Q16,served
@nb.njit(cache=True)
def block(arr,svc,pat,c):
    free=np.zeros(c); ab=0
    for i in range(arr.shape[0]):
        j=np.argmin(free); s=max(arr[i],free[j])
        if s-arr[i]>pat[i]: ab+=1
        else: free[j]=s+svc[i]
    return ab
def stream(rng,n,w,off,P):
    k=rng.poisson(n*w); arr=np.sort(np.concatenate([off+h*60+rng.uniform(0,60,x) for h,x in enumerate(k)]))
    m=len(arr); may=rng.random(m)<P['pm']
    sv=np.where(may,rng.triangular(*P['sw'],m),rng.triangular(*P['sr'],m))
    if P['shape']==1.0: pat=rng.exponential(P['pat'],m)
    else: pat=P['pat']/gamma(1+1/P['shape'])*rng.weibull(P['shape'],m)
    return arr,sv,pat,may
def block_ab(n,w,c,P,R=1500,seed=11):
    rng=np.random.default_rng(seed); A=T=0
    for _ in range(R):
        arr,sv,pt,my=stream(rng,n,w,0.0,P); A+=block(arr,sv,pt,c); T+=len(arr)
    return A/max(T,1)
def calibrate(target,w,P,lo,hi):
    for _ in range(16):
        mid=(lo+hi)/2
        if block_ab(mid,w,2,P,R=800)<target: lo=mid
        else: hi=mid
    return (lo+hi)/2
def run(P,R,seed0=0):
    n1,n2=P['n1'],P['n2']
    sh=(R,3,3,2,2)
    out={k:np.zeros(sh,np.int32) for k in ('X1','Lr','Lw','L1','L2','Q16','S')}; out['H']=np.zeros(sh)
    out['N1']=np.zeros((R,3),np.int32); out['N2']=np.zeros((R,3),np.int32)
    for r in range(R):
        ms=[stream(np.random.default_rng([seed0,r,1,a]),n1[a],wM,0.0,P) for a in range(3)]
        dl=np.random.default_rng([seed0,r,9]).uniform(5,10)
        af=[stream(np.random.default_rng([seed0,r,2,b]),n2[b],wA,T1,P) for b in range(3)]
        for a in range(3):
            out['N1'][r,a]=len(ms[a][0])
            for b in range(3):
                if a==0: out['N2'][r,b]=len(af[b][0])
                arr=np.concatenate([ms[a][0],af[b][0]]); sv=np.concatenate([ms[a][1],af[b][1]])
                pt=np.concatenate([ms[a][2],af[b][2]]); my=np.concatenate([ms[a][3],af[b][3]])
                for i,d1 in enumerate((2,3)):
                    for j,d2 in enumerate((2,3)):
                        x,lr,lw,h,l1,l2,q,s=day(arr,sv,pt,my,d1,d2,dl)
                        for k,v in (('X1',x),('Lr',lr),('Lw',lw),('H',h),('L1',l1),('L2',l2),('Q16',q),('S',s)): out[k][r,a,b,i,j]=v
    return out
