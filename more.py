import numpy as np, pickle
from scipy import stats
from ana import *
from core2 import block_ab, wM, wA, block
o={}
d=S['base']; Pp=d['P']
# (d) integrated vs isolated, 3-server contrast
rows=[]
for a in range(3):
    for i,c in ((0,2),(1,3)):
        iso=block_ab(Pp['n1'][a],wM,c,Pp,R=2000)
        L1=d['L1'][:,a,:,i,i].astype(float); N1=d['N1'][:,a][:,None]
        integ=L1.sum()/ (N1*np.ones((1,3))).sum()
        rows.append(('morning',a,c,round(iso,3),round(integ,3)))
for b in range(3):
    for i,c in ((0,2),(1,3)):
        iso=block_ab(Pp['n2'][b],wA,c,Pp,R=2000)
        # integrated: afternoon abandonment by arrivals in block 2, averaged over morning states with J0 conditional weights
        w=J0[:,b]/J0[:,b].sum()
        L2=np.array([d['L2'][:,a,b,i,i].sum() for a in range(3)]); N2=d['N2'][:,b].sum()
        integ=(w*L2).sum()/N2
        rows.append(('afternoon',b,c,round(iso,3),round(integ,3)))
for r in rows: print(r)
o['blocks']=rows
# aggregated 3-server morning abandonment (weighted by prior & arrivals)
pm=J0.sum(1); ag=[]
for i in (0,1):
    num=sum(pm[a]*d['L1'][:,a,0,i,i].mean() for a in range(3)); den=sum(pm[a]*d['N1'][:,a].mean() for a in range(3)); ag.append(num/den)
print('aggregate morning abandonment 2 serv %.3f 3 serv %.3f'%tuple(ag)); o['agg']=ag
# manager aggregate: weights by prior and volume
mg2=np.array([0.08,0.12,0.25]); mg3=np.array([0.05,0.06,0.07]); vol=Pp['n1']
print('manager aggregate 2: %.3f 3: %.3f'%((pm*vol*mg2).sum()/(pm*vol).sum(),(pm*vol*mg3).sum()/(pm*vol).sum()))
# congestion at 16:00
for i in (0,1):
    q=d['Q16'][:,:,0,i,0]; print('d1',i+2,'mean queue at 16:00 by state',q.mean(0).round(2),'P(q>0)',(q>0).mean(0).round(3))
o['q16']=[(d['Q16'][:,:,0,i,0].mean(0),(d['Q16'][:,:,0,i,0]>0).mean(0)) for i in (0,1)]
# (i) analytic posteriors N1 bins
n1=Pp['n1']; c=d['ncut']; cond=J0/pm[:,None]
for lo,hi in ((0,c[0]),(c[0]+1,c[1]),(c[1]+1,10**4)):
    lk=np.array([stats.poisson.cdf(hi,n)-stats.poisson.cdf(lo-1,n) for n in n1]); po=lk*pm; pk=po.sum(); po/=pk
    print('N1',lo,hi,'P',round(pk,3),'post th2',(po@cond).round(4))
xc=d['xcut'][1]; x=d['X1'][:,:,0,1,0]
for k,(lo,hi) in enumerate(((0,xc[0]),(xc[0]+1,xc[1]),(xc[1]+1,999))):
    lk=((x>=lo)&(x<=hi)).mean(0); po=lk*pm; pk=po.sum(); po/=pk; print('X1 d1=3',lo,hi,'P',round(pk,3),'post th2',(po@cond).round(4))
# (j) MC CI at v=500 (paired bootstrap, fixed policies)
pv=P(vr=500,vw=500); tc=TC(d,pv)[TE]; s=d['N1'][TE]; xx=d['X1'][TE][:,:,0,1,0]; rng=np.random.default_rng(1); b=[]
for _ in range(300):
    idx=rng.integers(0,2000,2000); t=tc[idx]
    st=(J0*t[...,1,1].mean(0)).sum()
    a1=(J0*((t[...,1,0]+(t[...,1,1]-t[...,1,0])*(s[idx][:,:,None]>=43)).mean(0))).sum()
    a2=(J0*((t[...,1,0]+(t[...,1,1]-t[...,1,0])*(xx[idx][:,:,None]>=5)).mean(0))).sum()
    b.append((st-a1,st-a2,a2-a1))
b=np.array(b); print('v500 CI N1',np.percentile(b[:,0],[2.5,97.5]).round(1),'X1',np.percentile(b[:,1],[2.5,97.5]).round(1),'diff',np.percentile(b[:,2],[2.5,97.5]).round(1))
o['ci500']=b
# (h) verification
lam,mu,gam,cc=1/4.0,1/6.0,1/15.0,2
K=400; p=np.zeros(K); p[0]=1
for n in range(1,K): p[n]=p[n-1]*lam/(mu*min(n,cc)+gam*max(n-cc,0))
p/=p.sum(); Pab=gam*sum(p[n]*max(n-cc,0) for n in range(K))/lam
rng=np.random.default_rng(5); T=2e6; n=rng.poisson(lam*T); arr=np.sort(rng.uniform(0,T,n))
sv=rng.exponential(1/mu,n); pt=rng.exponential(1/gam,n)
# batch means: 20 batches by arrival index
bs=[]; k=n//20
for j in range(20):
    sl=slice(j*k,(j+1)*k); a=arr[sl]-arr[sl][0]; bs.append(block(a,sv[sl],pt[sl],2)/k)
bs=np.array(bs); m=bs.mean(); h=stats.t.ppf(0.975,19)*bs.std(ddof=1)/np.sqrt(20)
print('ErlangA analytic %.4f sim %.4f CI [%.4f, %.4f] customers %d'%(Pab,m,m-h,m+h,n)); o['erlang']=(Pab,m,m-h,m+h,n)
cons=(d['S']+d['Lr']+d['Lw']==(d['N1'][:,:,None,None,None]+d['N2'][:,None,:,None,None])).all(); print('conservation',cons)
L=d['Lr']+d['Lw']; print('P(loss with 3/3 > 2/2 in same rep)',(L[...,1,1]>L[...,0,0]).mean().round(4),' P(2,3 > 2,2)',(L[...,0,1]>L[...,0,0]).mean().round(4))
pickle.dump(o,open('more.pkl','wb'))
