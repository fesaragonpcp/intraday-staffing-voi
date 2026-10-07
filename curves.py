import numpy as np, pickle
from ana import *
out={}
for sc in ('base','adjusted'):
    d=S[sc]
    for ca2 in (250.,150.):
        rows=[]
        for v in list(range(100,1001,50))+[1200,1500]:
            r=evaluate(d,P(vr=v,vw=v,ca2=ca2),J0)
            rows.append((v,r['static_pol'],round(r['V_perfect'],1),round(r['V_N1_thr'],1),round(r['V_X1_thr'],1),round(r['V_N1_free'],1),round(r['V_X1_free'],1),r['N1_thr_pol'],r['X1_thr_pol']))
        out[(sc,ca2)]=rows
        print('==',sc,'c_a2',ca2)
        for x in rows: print(x)
pickle.dump(out,open('curves.pkl','wb'))
