from __future__ import division
import numpy as np
from functions import rh
from ckm_2hdm import *
import matplotlib.pyplot as plt

hbar = 6.582119514e-25

tb = 37.9
mH = 1044
mu, mu_err = [0.00216,[0.00049,-0.00026]]
md, md_err = [0.00467,[0.00048,-0.00017]]
ms, ms_err = [0.093,[0.011,-0.005]]
mc, mc_err = [1.27,[0.02,-0.02]]
mb, mb_err = [4.18,[0.03,-0.02]]
mbpls, mbpls_err = [5.27933,[0.00013,-0.00013]]
mdpls, mdpls_err = [1.86965,[0.00005,-0.00005]]
mdspls, mdspls_err  = [1.96834,[0.00007,-0.00007]]
mtau, mtau_err = [1.77686,[0.00012,-0.00012]]
mmu, mmu_err = [0.105658374,[0.0000000024,-0.0000000024]]
mK, mK_err = [0.493677,[0.000016,-0.000016]]
mpi, mpi_err = [0.13957031,[0.00000024,-0.00000024]]
fb, fb_err = [0.190,[0.0013,-0.0013]]
fd, fd_err = [0.212,[0.0007,-0.0007]]
fds, fds_err = [0.2499,[0.0005,-0.0005]]
fKpi, fKpi_err = [1.1932,[0.0019,-0.0019]]
delt_kpi, delt_kpi_err = [-0.0069,[0.0017,-0.0017]]
delt_tau, delt_tau_err = [0.0003,[0,0]]
taub, taub_err = [(1.638e-12)/hbar,[(0.004e-12)/hbar,-(0.004e-12)/hbar]]
taud, taud_err = [(1040e-15)/hbar,[(7e-15)/hbar,-(7e-15)/hbar]]
tauds, tauds_err = [(504e-15)/hbar,[(4e-15)/hbar,-(4e-15)/hbar]]
brb, brb_err = [1.09e-4,[0.24e-4,-0.24e-4]]
brd, brd_err = [3.82e-4,[0.33e-4,-0.33e-4]]
brds, brds_err = [0.0548,[0.0023,-0.0023]]
brk, brk_err = [1.337,[0.0032,-0.0032]]
brtk, brtk_err = [0.34,[0.03,-0.03]]

### PDG
Vud,Vud_err = [0.97420,[0.00021,-0.00021]]
Vus,Vus_err = [0.2243,[0.0005,-0.0005]]
Vub,Vub_err = [0.00401,[0.00037,-0.00037]]
Vcd,Vcd_err = [0.2164,[0.0052,-0.0052]]
#Vcs,Vcs_err = [0.997,[0.017,-0.017]]
Vcs,Vcs_err = [1.006,[0.019,-0.019]]
Vcb,Vcb_err = [0.0422,[0.0008,-0.0008]]
Vtd,Vtd_err = [0.0081,[0.0005,-0.0005]]
Vts,Vts_err = [0.0394,[0.0023,-0.0023]]
Vtb,Vtb_err = [1.019,[0.025,-0.025]]

#h1,t1,v1,h2,t2,v2,ha,ta,va = ckmelsr([Vud,Vus,Vub,Vcd,Vcs,Vcb],[Vud_err,Vus_err,Vub_err,Vcd_err,Vcs_err,Vcb_err],mu,mu_err,md,md_err,ms,ms_err,mc,mc_err,mb,mb_err,mbpls,mbpls_err,mdpls,mdpls_err,mdspls,mdspls_err,mtau,mtau_err,mmu,mmu_err,fb,fb_err,fd,fd_err,fds,fds_err,taub,taub_err,taud,taud_err,tauds,tauds_err,brb,brb_err,brd,brd_err,brds,brds_err,mK,mK_err,mpi,mpi_err,fKpi,fKpi_err,delt_kpi,delt_kpi_err,delt_tau,delt_tau_err,brk,brk_err,brtk,brtk_err)
#
#print len(h1)
#print len(h2)
#print len(ha)
#
#plt.figure(figsize=(8,6))
#plt.scatter(t1,h1,c='green')
#plt.scatter(t2,h2,c='cornflowerblue')
#plt.scatter(ta,ha,c='darkorchid')
#plt.axis([-1,2,0,3.5])
#plt.title('SM4 Allowed Unitarity with 2HDM',fontsize=18)
#plt.xlabel('$\\log[\\tan(\\beta)]$',fontsize=18)
#plt.ylabel('$\\log[m_{H+}]$',fontsize=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.show()
#quit()
### CKMfitter
#Vud,Vud_err = [0.974390,[0.000014,-0.000058]]
#Vus,Vus_err = [0.224834,[0.000252,-0.000059]]
#Vub,Vub_err = [0.003683,[0.000075,-0.000061]]
#Vcd,Vcd_err = [0.224701,[0.000254,-0.000058]]
#Vcs,Vcs_err = [0.973539,[0.000038,-0.000060]]
#Vcb,Vcb_err = [0.041620,[0.000260,-0.000800]]
#Vtd,Vtd_err = [0.008545,[0.000075,-0.000157]]
#Vts,Vts_err = [0.040900,[0.000260,-0.000760]]
#Vtb,Vtb_err = [0.999127,[0.000032,-0.000012]]

def rprop(V,V_err,mu,mu_err,md,md_err,mm,mm_err):
    ex = V/pow((1+rh(mu,md,mm,tb,mH)),2)
    u1 = abs(((V+V_err[0])/pow((1+rh(mu,md,mm,tb,mH)),2))-ex)
    l1 = abs(((V+V_err[1])/pow((1+rh(mu,md,mm,tb,mH)),2))-ex)
    u2 = abs((V/pow((1+rh(mu+mu_err[0],md,mm,tb,mH)),2))-ex)
    l2 = abs((V/pow((1+rh(mu+mu_err[1],md,mm,tb,mH)),2))-ex)
    u3 = abs((V/pow((1+rh(mu,md+md_err[0],mm,tb,mH)),2))-ex)
    l3 = abs((V/pow((1+rh(mu,md+md_err[1],mm,tb,mH)),2))-ex)
    u4 = abs((V/pow((1+rh(mu,md,mm+mm_err[0],tb,mH)),2))-ex)
    l4 = abs((V/pow((1+rh(mu,md,mm+mm_err[1],tb,mH)),2))-ex)

    ups = np.sqrt(u1**2 + u2**2 + u3**2 + u4**2)
    ls = np.sqrt(l1**2 + l2**2 + l3**2 + l4**2)

    return ups, ls

#Vubn = Vub/pow((1+rh(mu,mb,mbpls,tb,mH)),2)
#Vcdn = Vcd/pow((1+rh(mc,md,mdpls,tb,mH)),2)
#Vcsn = Vcs/pow((1+rh(mc,ms,mdspls,tb,mH)),2)
#print Vubn, Vub/Vubn
#print Vcdn, Vcd/Vcdn
#print Vcsn, Vcs/Vcsn
#
#Vubn_err = rprop(Vub,Vub_err,mu,mu_err,mb,mb_err,mbpls,mbpls_err)
#Vcdn_err = rprop(Vcd,Vcd_err,mc,mc_err,md,md_err,mdpls,mdpls_err)
#Vcsn_err = rprop(Vcs,Vcs_err,ms,ms_err,ms,ms_err,mdspls,mdspls_err)
#
#Vubp2 = Vp2([Vud,Vus,Vubn],[Vud_err,Vus_err,Vubn_err])
#print "|Vub'|^2 =(",Vubp2[0]*1e4,"+",Vubp2[1]*1e4,"-",Vubp2[2]*1e4,")* e-4"
##print np.sqrt(Vubp2[0])
#
Vcbp2 = Vp2([Vcd,Vcs,42.22e-3],[Vcd_err,Vcs_err,[0.60e-3,-0.62e-3]])
print "|Vcb'|^2 =(",Vcbp2[0]*1e2,"+",Vcbp2[1]*1e2,"-",Vcbp2[2]*1e2,")* e-2"
#print np.sqrt(abs(Vcbp2[0]))
#
#Vtbp2 = Vp2([Vubn,Vcb,Vtb],[Vubn_err,Vcb_err,Vtb_err])
#print "|Vt'b|^2 =(",Vtbp2[0]*1e2,"+",Vtbp2[1]*1e2,"-",Vtbp2[2]*1e2,")* e-2"
##print np.sqrt(abs(Vtbp2[0]))
#
#Vtdp2 = Vp2([Vud,Vcdn],[Vud_err,Vcdn_err])
#Vtsp2 = Vp2([Vus,Vcsn],[Vus_err,Vcsn_err])
#print "Vtd^2 + Vt'd^2 =(",Vtdp2[0]*1e3,"+",Vtdp2[1]*1e3,"-",Vtdp2[2]*1e3,")* e-3"
#print "Vts^2 + Vt's^2 =(",Vtsp2[0]*1e2,"+",Vtsp2[1]*1e2,"-",Vtsp2[2]*1e2,")* e-2"

