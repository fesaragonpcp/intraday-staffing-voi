import numpy as np, pickle
S=pickle.load(open('sims.pkl','rb'))
J0=np.array([[0.14,0.06,0.0],[0.15,0.15,0.0],[0.10,0.10,0.30]])
TR=slice(0,2000); TE=slice(2000,4000)
BASE=dict(ca1=120.,ca2=250.,co=85.,vr=87.5,vw=600.)
def P(**kw): p=dict(BASE); p.update(kw); return p
def TC(d,p):
    i=np.array([0,1])[None,None,None,:,None]; j=np.array([0,1])[None,None,None,None,:]
    return p['ca1']*i*8+p['ca2']*j*6+p['co']*(2+j)*d['H']+p['vr']*d['Lr']+p['vw']*d['Lw']
def ncuts(n):
    return tuple(int(np.floor((n[k+1]-n[k])/np.log(n[k+1]/n[k]))) for k in (0,1))
def xcuts(d):
    out={}
    for i in (0,1):
        x=d['X1'][:,:,0,i,0]; M=x.max()+2; pm=[np.bincount(x[:,a],minlength=M)/x.shape[0] for a in range(3)]
        lo=next((v for v in range(M) if pm[1][v]>pm[0][v]),1); hi=next((v for v in range(M) if pm[2][v]>pm[1][v]),lo+1)
        out[i]=(lo-1,hi-1)
    return out
for k in S: S[k]['xcut']=xcuts(S[k]); S[k]['ncut']=ncuts(S[k]['P']['n1'])
def sig(d,name,i): return d['N1'] if name=='N1' else d['X1'][:,:,0,i,0]
def binz(d,name,i,s):
    c=d['ncut'] if name=='N1' else d['xcut'][i]
    return np.where(s<=c[0],0,np.where(s<=c[1],1,2))
def wmean(J,a): return (J*a.mean(0)).sum()
def evaluate(d,p,J):
    tc=TC(d,p); tr=tc[TR]; te=tc[TE]; L=(d['Lr']+d['Lw']); Lte=L[TE]
    R={}
    st={(i,j):wmean(J,tr[...,i,j]) for i in (0,1) for j in (0,1)}
    i0,j0=min(st,key=st.get); R['static']=wmean(J,te[...,i0,j0]); R['static_pol']=(i0+2,j0+2); R['static_L']=wmean(J,Lte[...,i0,j0])
    R['static_all']={(i+2,j+2):(round(wmean(J,te[...,i,j]),1),round(wmean(J,Lte[...,i,j]),2)) for i in (0,1) for j in (0,1)}
    def apply(i,choice):  # choice[r,a,b] in {0,1} on test
        c=te[...,i,0]+(te[...,i,1]-te[...,i,0])*choice; l=Lte[...,i,0]+(Lte[...,i,1]-Lte[...,i,0])*choice
        return wmean(J,c),wmean(J,l)
    # EVPI theta2
    best=(1e18,)
    for i in (0,1):
        m=tr[...,i,:].mean(0); c=(J*m.min(-1)).sum()
        if c<best[0]: best=(c,i,m.argmin(-1))
    _,i,jj=best; ch=np.broadcast_to(jj[None],te.shape[:3]).astype(float)
    R['perfect'],R['perfect_L']=apply(i,ch)
    # EVPI state + congestion at 16:00 (Q16 bins 0,1,2,3+)
    best=(1e18,)
    for i in (0,1):
        q=np.minimum(d['Q16'][:,:,:,i,0],3); qt=q[TR]; tot=0; pol=np.zeros((3,3,4),int)
        for v in range(4):
            m=(qt==v)
            c0=(J*(tr[...,i,0]*m).mean(0)); c1=(J*(tr[...,i,1]*m).mean(0)); pol[...,v]=(c1<c0); tot+=np.minimum(c0,c1).sum()
        if tot<best[0]: best=(tot,i,pol)
    _,i,pol=best; q=np.minimum(d['Q16'][:,:,:,i,0],3)[TE]
    a_=np.arange(3)[None,:,None]; b_=np.arange(3)[None,None,:]
    ch=pol[a_,b_,q]; R['perfectQ'],_=apply(i,ch)
    for name in ('N1','X1'):
        # threshold
        best=(1e18,)
        for i in (0,1):
            s=sig(d,name,i)[TR]; diff=tr[...,i,1]-tr[...,i,0]; b0=wmean(J,tr[...,i,0])
            for t in list(np.unique(s))+[s.max()+1]:
                c=b0+wmean(J,diff*(s[:,:,None]>=t))
                if c<best[0]: best=(c,i,int(t))
        _,i,t=best; s=sig(d,name,i)[TE]; ch=(s[:,:,None]>=t)*np.ones(3)[None,None,:]
        R[name+'_thr'],R[name+'_thr_L']=apply(i,ch); R[name+'_thr_pol']=(i+2,t)
        R[name+'_trig']=(J.sum(1)*(s>=t).mean(0)).sum()
        # free per count
        best=(1e18,)
        for i in (0,1):
            s=sig(d,name,i)[TR]; diff=tr[...,i,1]-tr[...,i,0]; b0=wmean(J,tr[...,i,0]); pol={}; c=b0
            for v in np.unique(s):
                g=wmean(J,diff*(s[:,:,None]==v)); pol[v]=g<0; c+=min(g,0)
            if c<best[0]: best=(c,i,pol)
        _,i,pol=best; s=sig(d,name,i)[TE]; dflt=st[(i,1)]<st[(i,0)]
        ch=np.vectorize(lambda v: pol.get(v,dflt))(s).astype(float)[:,:,None]*np.ones(3)
        R[name+'_free'],_=apply(i,ch)
        # bins
        best=(1e18,)
        for i in (0,1):
            bb=binz(d,name,i,sig(d,name,i))[TR]; diff=tr[...,i,1]-tr[...,i,0]; b0=wmean(J,tr[...,i,0]); c=b0; pol=[]
            for k in range(3):
                g=wmean(J,diff*(bb[:,:,None]==k)); pol.append(int(g<0)); c+=min(g,0)
            if c<best[0]: best=(c,i,pol)
        _,i,pol=best; bb=binz(d,name,i,sig(d,name,i))[TE]; ch=np.array(pol)[bb][:,:,None]*np.ones(3)
        R[name+'_bins'],_=apply(i,ch); R[name+'_bins_pol']=(i+2,pol)
    for k in ('perfect','perfectQ','N1_thr','N1_free','N1_bins','X1_thr','X1_free','X1_bins'): R['V_'+k]=R['static']-R[k]
    return R
if __name__=='__main__':
    d=S['base']; r=evaluate(d,BASE,J0)
    print('ncut',d['ncut'],'xcut',d['xcut'])
    for k,v in r.items(): print(k, np.round(v,2) if isinstance(v,float) else v)
