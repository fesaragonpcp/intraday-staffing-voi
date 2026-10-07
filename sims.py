import numpy as np, pickle, time
from core2 import *
B=dict(n1=np.array([15.,30.,60.]),n2=np.array([10.,25.,40.]),pm=0.2,sr=(10,15,20),sw=(30,40,50),pat=15.0,shape=1.0)
sets={}
sets['base']=dict(B)
adj=dict(B); adj['n1']=np.array([15.,30.,calibrate(0.25,wM,B,30,60)]); adj['n2']=np.array([10.,25.,calibrate(0.25,wA,B,20,40)]); sets['adjusted']=adj
for k,ch in (('pat10',dict(pat=10.0)),('pat20',dict(pat=20.0)),('weib2',dict(shape=2.0)),('may10',dict(pm=0.1)),('may30',dict(pm=0.3))):
    p=dict(B); p.update(ch); sets[k]=p
res={}
for k,p in sets.items():
    t=time.time(); res[k]=run(p,4000); res[k]['P']=p; print(k,p['n1'].round(1),p['n2'].round(1),round(time.time()-t,1),'s',flush=True)
pickle.dump(res,open('sims.pkl','wb'))
