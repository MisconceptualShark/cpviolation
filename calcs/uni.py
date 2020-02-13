from __future__ import division
import numpy as np

Vud,Vud_err = [0.974390,[0.000014,-0.000058]]
Vus,Vus_err = [0.224834,[0.000252,-0.000059]]
Vub,Vub_err = [0.003683,[0.000075,-0.000061]]
Vcd,Vcd_err = [0.224701,[0.000254,-0.000058]]
Vcs,Vcs_err = [0.973539,[0.000038,-0.000060]]
Vcb,Vcb_err = [0.04162,[0.00026,-0.00080]]
Vtd,Vtd_err = [0.008545,[0.000075,-0.000157]]
Vts,Vts_err = [0.04090,[0.00026,-0.00076]]
Vtb,Vtb_err = [0.999127,[0.000032,-0.000012]]

def v2(vs):
    v = 1
    for i in range(len(vs)):
        v -= vs[i]**2
    return v

def Vp2(vs,vse):
    v = v2(vs)
    u,l=0,0
    for i in range(len(vs)):
        u += pow((2*vse[i][0])/vs[i],2)
        l += pow((2*vse[i][1])/vs[i],2)
#        vs[i] += vse[i][0]
#        u += abs(v2(vs)-v)**2
#        vs[i] += (vse[i][1]-vse[i][0])
#        l += abs(v2(vs)-v)**2
#        vs[i] -= vse[i][1]
    u = v*np.sqrt(u)
    l = v*np.sqrt(l)
    return v, u, l


Vubp2 = Vp2([Vud,Vus,Vub],[Vud_err,Vus_err,Vub_err])
print "|Vub'|^2 =(",Vubp2[0]*1e7,"+",Vubp2[1]*1e7,"-",Vubp2[2]*1e7,")* e-7"

Vcbp2 = Vp2([Vcd,Vcs,Vcb],[Vcd_err,Vcs_err,Vcb_err])
print "|Vcb'|^2 =(",Vcbp2[0]*1e7,"+",Vcbp2[1]*1e7,"-",Vcbp2[2]*1e7,")* e-7"

Vtbp2 = Vp2([Vub,Vcb,Vtb],[Vub_err,Vcb_err,Vtb_err])
print "|Vt'b|^2 =(",Vtbp2[0]*1e7,"+",Vtbp2[1]*1e7,"-",Vtbp2[2]*1e7,")* e-7"

Vtdp2 = Vp2([Vud,Vcd],[Vud_err,Vcd_err])
Vtsp2 = Vp2([Vus,Vcs],[Vus_err,Vcs_err])
print "Vtd^2 + Vt'd^2 =(",Vtdp2[0]*1e5,"+",Vtdp2[1]*1e5,"-",Vtdp2[2]*1e5,")* e-5"
print "Vts^2 + Vt's^2 =(",Vtsp2[0]*1e3,"+",Vtsp2[1]*1e3,"-",Vtsp2[2]*1e3,")* e-3"

