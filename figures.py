"""Figures 1-3 of the manuscript. Run after sims.py, all.py and grid.py."""
import pickle, subprocess, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8})
subprocess.run(['dot','-Tpng','-Gdpi=300','influence_diagram.dot','-o','fig1.png'],check=True)
V,C,G=pickle.load(open('grid.pkl','rb')); V=np.array(V); C=np.array(C)
f,a=plt.subplots(figsize=(3.35,2.7),dpi=300)
im=a.pcolormesh(np.r_[V-25,V[-1]+25],np.r_[C-20,C[-1]+20],G,cmap='Greys',shading='flat',vmin=0,vmax=600)
cb=f.colorbar(im,ax=a,pad=0.02); cb.set_label('EVSI of $N_1$ (MXN/day)',fontsize=7); cb.ax.tick_params(labelsize=6)
a.errorbar([190],[250],yerr=[[100],[150]],fmt='o',color='k',mfc='white',ms=4,capsize=2,lw=0.8); a.text(205,262,'manager\n(margin)',fontsize=6)
a.plot([150],[250],'^',color='k',mfc='white',ms=4); a.text(95,228,'WTP',fontsize=6)
a.set_xlabel('Value of a lost customer (MXN)'); a.set_ylabel('Afternoon opportunity cost, $c_{a,2}$ (MXN/h)')
f.tight_layout(); f.savefig('fig2.png')
pers=pickle.load(open('all.pkl','rb'))['pers']; cf=FuncFormatter(lambda v,p: f'{v:.1f}'.replace('.',','))
f,a=plt.subplots(figsize=(3.35,2.5),dpi=300)
for path,ls,lab in (('max','-','to maximal persistence'),('elic','--','to elicited prior')):
    r=[x for x in pers if x[0]==path]; c=[x[2] for x in r]
    a.plot(c,[max(0,x[5]) for x in r],'k'+ls,marker='o',ms=2.5,label='$N_1$, '+lab)
    a.plot(c,[max(0,x[6]) for x in r],ls,color='0.55',marker='s',ms=2.5,label='$X_1$, '+lab)
a.axhline(510,color='k',ls=':',lw=0.8); a.text(0.82,520,'EVPI',fontsize=6.5,ha='right')
a.xaxis.set_major_formatter(cf); a.set_xlabel('Morning–afternoon state correlation'); a.set_ylabel('Value per day (MXN)')
a.legend(frameon=False,fontsize=6,loc='lower right'); a.spines[['top','right']].set_visible(False); a.set_ylim(0,580)
f.tight_layout(); f.savefig('fig3.png')
