import numpy as np, pickle
from ana import *
out={}
d=S['base']
# (e) valuation comparison
val=[]
for lab,p in (('Margin 87.5/600',P()),('Uniform 190',P(vr=190,vw=190)),('Uniform 150 (WTP)',P(vr=150,vw=150)),('Lost sales 350/4000',P(vr=350,vw=4000)),('Uniform 500 (illustrative)',P(vr=500,vw=500))):
    r=evaluate(d,p,J0)
    val.append((lab,r['static_pol'],round(r['static'],0),round(r['static_L'],2),r['N1_thr_pol'],round(r['N1_thr'],0),round(r['N1_thr_L'],2),round(r['V_N1_thr'],1),round(r['V_X1_thr'],1),round(r['V_perfect'],1),round(r['V_perfectQ'],1),round(r['V_N1_free'],1),round(r['V_X1_free'],1),round(r['V_N1_bins'],1),round(r['V_X1_bins'],1),round(r['N1_trig'],3)))
for x in val: print(x)
out['val']=val
print('static_all base', evaluate(d,P(),J0)['static_all'])
print('static_all v500', evaluate(d,P(vr=500,vw=500),J0)['static_all'])
# (a) persistence at v=500
pv=P(vr=500,vw=500)
PM=J0.sum(1); PA=J0.sum(0); Ji=np.outer(PM,PA)
def comon(pm,pa):
    J=np.zeros((3,3)); a=pm.copy(); b=pa.copy(); i=j=0
    while i<3 and j<3:
        m=min(a[i],b[j]); J[i,j]+=m; a[i]-=m; b[j]-=m
        if a[i]<=1e-12: i+=1
        if b[j]<=1e-12: j+=1
    return J
def corr(J):
    s=np.arange(3); mx=(J.sum(1)*s).sum(); my=(J.sum(0)*s).sum()
    return (J*np.outer(s-mx,s-my)).sum()/np.sqrt((J.sum(1)*(s-mx)**2).sum()*(J.sum(0)*(s-my)**2).sum())
Jc=comon(PM,PA); out['Jc']=Jc; out['Ji']=Ji; print('Jmax',Jc.round(3)); print('Jind',Ji.round(3))
pers=[]
for path,Jend in (('max',Jc),('elic',J0)):
    for a in (0,0.25,0.5,0.75,1.0):
        J=(1-a)*Ji+a*Jend; r=evaluate(d,pv,J)
        pers.append((path,a,round(corr(J),3),round(r['V_perfect'],1),round(r['V_perfectQ'],1),round(r['V_N1_thr'],1),round(r['V_X1_thr'],1)))
for x in pers: print(x)
out['pers']=pers
# (c) independence: free policies
r=evaluate(d,pv,Ji); print('indep v500: EVPI',round(r['V_perfect'],1),'EVPI+Q',round(r['V_perfectQ'],1),'N1 free',round(r['V_N1_free'],1),'X1 free',round(r['V_X1_free'],1))
out['indep']=(r['V_perfect'],r['V_perfectQ'],r['V_N1_free'],r['V_X1_free'],r['V_N1_thr'],r['V_X1_thr'])
# (f) sensitivity direct
sens=[]
for k in ('base','adjusted','pat10','pat20','weib2','may10','may30'):
    for lab,p in (('margin',P()),('v500',pv)):
        r=evaluate(S[k],p,J0)
        sens.append((k,lab,r['static_pol'],r['N1_thr_pol'],round(r['V_N1_thr'],1),round(r['V_X1_thr'],1),round(r['V_perfect'],1)))
for lab,p in (('ca2=150 margin',P(ca2=150.)),('ca2=400 v500',P(vr=500,vw=500,ca2=400.)),('ca2=150 v500',P(vr=500,vw=500,ca2=150.)),('ca1=80 margin',P(ca1=80.)),('ca1=200 margin',P(ca1=200.))):
    r=evaluate(d,p,J0); sens.append(('base',lab,r['static_pol'],r['N1_thr_pol'],round(r['V_N1_thr'],1),round(r['V_X1_thr'],1),round(r['V_perfect'],1)))
for x in sens: print(x)
out['sens']=sens
pickle.dump(out,open('all.pkl','wb'))
