from __future__ import division
import numpy as np

Vud,Vud_err = [0.97420,[0.00021,-0.00021]]
Vus,Vus_err = [0.2243,[0.0005,-0.0005]]
Vub,Vub_err = [0.00394,[0.00036,-0.00036]]
Vcd,Vcd_err = [0.218,[0.004,-0.004]]
Vcs,Vcs_err = [0.997,[0.017,-0.017]]
Vcb,Vcb_err = [0.0422,[0.0008,-0.0008]]
Vtd,Vtd_err = [0.0081,[0.0005,-0.0005]]
Vts,Vts_err = [0.0394,[0.0023,-0.0023]]
Vtb,Vtb_err = [1.019,[0.025,-0.025]]

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
print "|Vub'|^2 =(",Vubp2[0]*1e4,"+",Vubp2[1]*1e4,"-",Vubp2[2]*1e4,")* e-4"
print np.sqrt(Vubp2[0])

Vcbp2 = Vp2([Vcd,Vcs,Vcb],[Vcd_err,Vcs_err,Vcb_err])
print "|Vcb'|^2 =(",Vcbp2[0]*1e2,"+",Vcbp2[1]*1e2,"-",Vcbp2[2]*1e2,")* e-2"
print np.sqrt(abs(Vcbp2[0]))

Vtbp2 = Vp2([Vub,Vcb,Vtb],[Vub_err,Vcb_err,Vtb_err])
print "|Vt'b|^2 =(",Vtbp2[0]*1e2,"+",Vtbp2[1]*1e2,"-",Vtbp2[2]*1e7,")* e-2"
print np.sqrt(abs(Vtbp2[0]))

Vtdp2 = Vp2([Vud,Vcd],[Vud_err,Vcd_err])
Vtsp2 = Vp2([Vus,Vcs],[Vus_err,Vcs_err])
print "Vtd^2 + Vt'd^2 =(",Vtdp2[0]*1e2,"+",Vtdp2[1]*1e2,"-",Vtdp2[2]*1e2,")* e-2"
print "Vts^2 + Vt's^2 =(",Vtsp2[0]*1e2,"+",Vtsp2[1]*1e2,"-",Vtsp2[2]*1e2,")* e-2"

