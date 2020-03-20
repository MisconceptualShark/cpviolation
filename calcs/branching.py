from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from functions import *
from rdstarring import *
from fitting import *
#from ckm_2hdm import *
import os 

g_gev = (1.1663787e-5)**2
hbar_gev = 6.582119514e-25
g_mev = (1.1663787e-11)**2
hbar_mev = 6.582119514e-22

################ LEPTONIC AND SEMILEPTONIC DECAY

m_bplus, m_bplus_err = [5.27933,[0.00013,-0.00013]]
m_dplus08, m_dplus08_err = [1.86962,[0.00020,-0.00020]] #08
m_dplus, m_dplus_err = [1.86965,[0.00005,-0.00005]]
m_dstar, m_dstar_err = [2.1122,[0.0004,-0.0004]]
m_dsplus, m_dsplus_err = [1.96834,[0.00007,-0.00007]]
m_B08, m_B08_err = [5.27953,[0.00033,-0.00033]] #08
m_Brd, m_Brd_err = [5.27964,[0.00013,-0.00013]]
m_K, m_K_err = [0.493677,[0.000016,-0.000016]]
m_pi, m_pi_err = [0.13957061,[0.00000024,-0.00000024]]
m_tau, m_tau_err = [1.77686,[0.00012,-0.00012]]
m_mu, m_mu_err = [0.1056583745,[0.0000000024,-0.0000000024]]
m_u, m_u_err = [0.00216,[0.00049,-0.00026]]
m_d, m_d_err = [0.00467,[0.00048,-0.00017]]
m_s, m_s_err = [0.093,[0.011,-0.005]]
m_c08, m_c08_err = [1.27,[0.07,-0.11]] #08
m_c, m_c_err = [1.27,[0.02,-0.02]]
m_b08, m_b08_err = [4.20,[0.17,-0.07]] #08
m_b, m_b_err = [4.18,[0.03,-0.02]]

Vud, Vud_err = [0.974390,[0.000014,-0.000058]]
Vus, Vus_err = [0.224834,[0.000252,-0.000059]]
Vub, Vub_err = [0.003683,[0.000075,-0.000061]]
Vcd, Vcd_err = [0.224701,[0.000254,-0.000058]]
Vcs, Vcs_err = [0.973539,[0.000038,-0.000060]]
Vcb, Vcb_err = [0.04162,[0.00026,-0.00080]]
Vtd, Vtd_err = [0.008545,[0.000075,-0.000157]]
Vts, Vts_err = [0.04090,[0.00026,-0.00076]]
Vtb, Vtb_err = [0.999127,[0.000032,-0.000012]]

f_bplus, f_bplus_err = [0.190,[0.0013,-0.0013]]
f_dplus, f_dplus_err = [0.212,[0.0007,-0.0007]]
f_dsplus, f_dsplus_err = [0.2499,[0.0005,-0.0005]]
f_Kpi, f_Kpi_err = [1.1932,[0.0019,-0.0019]]

delt_kpi, delt_kpi_err = [-0.0069,[0.0017,-0.0017]]
delt_kpitau, delt_kpitau_err = [0.0003,[0,0]]

rho08, rho08_err = [1.19,[0.057,-0.057]] #08
rho, rho_err = [1.131,[0.033,-0.033]] #1909.12524
delt_rd, delt_rd_err = [0.46,[0.01,-0.01]] #0802.3790

tau_bplus, tau_bplus_err = [(1.638e-12)/hbar_gev,[(0.004e-12)/hbar_gev,-(0.004e-12)/hbar_gev]]
tau_b, tau_berr = [(1.519e-12)/hbar_gev,[(0.004e-12)/hbar_gev,-(0.004e-12)/hbar_gev]]
tau_dplus, tau_dplus_err = [(1040e-15)/hbar_gev,[(7e-15)/hbar_gev,-(7e-15)/hbar_gev]]
tau_dsplus, tau_dsplus_err = [(504e-15)/hbar_gev,[(4e-15)/hbar_gev,-(4e-15)/hbar_gev]]

bplus_exp, bplus_err_exp = [1.09e-4,[0.24e-4,-0.24e-4]]
dplus_exp, dplus_err_exp = [3.77e-4,[0.17e-4,-0.17e-4]]
dsplus_exp, dsplus_err_exp = [0.0548,[0.0023,-0.0023]]
bplusmu_exp, bplusmu_err_exp = [5.3e-7,[2.2e-7,-2.2e-7]]
dsmu_exp, dsmu_err_exp = [5.52e-3,[0.16e-3,-0.16e-3]]
kpi_exp, kpi_exp_err = [1.337,[0.0032,-0.0032]]
kpitau_exp, kpitau_exp_err = [6.438e-2,[9.384e-4,-9.384e-4]]
rd08_exp, rd08_exp_err = [0.416,[0.128,-0.128]] #08
rd_exp, rd_exp_err = [0.34,[0.03,-0.03]] #1909.12524

rhod, rhod_err = [1.122,[0.024,-0.024]]
delta, delta_err = [1,[1,-1]]
r01, r01_err = [1.85,[0.05,-0.05]]
r11, r11_err = [1.270,[0.026,-0.026]]
r21, r21_err = [0.852,[0.018,-0.018]]
vev, vev_err = [246,[0,0]]
rdst_exp, rdst_exp_err = [0.295,[0.014,-0.014]]

delta_b,delta_d = 0.982,0.99*0.982

#bdle = (5.05e-2)/tau_b
#bdle_err = bdle*np.sqrt((0.0014/0.0505)**2 + (tau_berr[0]/tau_b)**2)

#print error_rds(m_Brd,m_Brd_err,m_dstar,m_dstar_err,rhod,rhod_err,r01,r01_err,r11,r11_err,r21,r21_err,Vcb,Vcb_err,m_mu,m_mu_err,m_tau,m_tau_err,vev,vev_err,m_c,m_c_err,m_b,m_b_err,1,1)
#print error_rdn(m_Brd,m_Brd_err,m_dplus,m_dplus_err,rho,rho_err,delta,delta_err,Vcb,Vcb_err,m_mu,m_mu_err,m_tau,m_tau_err,vev,vev_err,m_c,m_c_err,m_b,m_b_err,46,1172)
#quit()

#bmu, bmerr = bsm(m_bplus,m_mu,Vub,f_bplus,tau_bplus,1), error_branching(m_bplus,m_bplus_err,m_mu,m_mu_err,Vub,Vub_err,f_bplus,f_bplus_err,tau_bplus,tau_bplus_err,1,[1,1],1,[1,1],1,1,1)
#btau, berr = bsm(m_bplus,m_tau,Vub,f_bplus,tau_bplus,1), error_branching(m_bplus,m_bplus_err,m_tau,m_tau_err,Vub,Vub_err,f_bplus,f_bplus_err,tau_bplus,tau_bplus_err,1,[1,1],1,[1,1],1,1,1)
#dtau, derr = bsm(m_dplus,m_mu,Vcd,f_dplus,tau_dplus,0.982*0.99), error_branching(m_dplus,m_dplus_err,m_mu,m_mu_err,Vcd,Vcd_err,f_dplus,f_dplus_err,tau_dplus,tau_dplus_err,1,[1,1],1,[1,1],1,1,0.982*0.99)
#dsmu, dsmerr = bsm(m_dsplus,m_mu,Vcs,f_dsplus,tau_dsplus,1), error_branching(m_dsplus,m_dsplus_err,m_mu,m_mu_err,Vcs,Vcs_err,f_dsplus,f_dsplus_err,tau_dsplus,tau_dsplus_err,1,[1,1],1,[1,1],1,1,1)
#dstau, dserr = bsm(m_dsplus,m_tau,Vcs,f_dsplus,tau_dsplus,delta_b), error_branching(m_dsplus,m_dsplus_err,m_tau,m_tau_err,Vcs,Vcs_err,f_dsplus,f_dsplus_err,tau_dsplus,tau_dsplus_err,1,[1,1],1,[1,1],1,1,delta_b)
#print("B -> tau nu Branching =",btau*1e5,"+",berr[0]*1e5,"-",berr[1]*1e5)
#print("B -> mu nu Branching =",bmu*1e7,"+",bmerr[0]*1e7,"-",bmerr[1]*1e7)
#print("D -> mu nu Branching =",dtau*1e4,"+",derr[0]*1e4,"-",derr[1]*1e4)
#print("Ds -> tau nu Branching =",dstau*1e2,"+",dserr[0]*1e2,"-",dserr[1]*1e2)
#print("Ds -> mu nu Branching =",dsmu*1e3,"+",dsmerr[0]*1e3,"-",dsmerr[1]*1e3)
#
#kpi, tkpi = decay_ratios(m_K,m_pi,m_mu,m_tau,Vus,Vud,f_Kpi,delt_kpi,delt_kpitau)
#kerr0, kerr1, terr0, terr1 = error_kpi(m_K,m_K_err,m_pi,m_pi_err,m_mu,m_mu_err,m_tau,m_tau_err,Vus,Vus_err,Vud,Vud_err,f_Kpi,f_Kpi_err,delt_kpi,delt_kpi_err,delt_kpitau,delt_kpitau_err,m_s,m_s_err,m_d,m_d_err,m_u,m_u_err,1,1)
#print("K/pi branching",kpi,"+",kerr0,"-",kerr1)
#print("tau -> K/pi branching",tkpi*1e2,"+",terr0*1e2,"-",terr1*1e2)
#quit()

# OLD INPUTS
#m_bplus, m_bplus_err = [5.27917,[0.00029,-0.00029]]
#m_dplus, m_dplus_err = [1.86962,[0.00020,-0.00020]] #08
#m_dsplus, m_dsplus_err = [1.96849,[0.00034,-0.00034]]
#m_Brd, m_Brd_err = [5.27953,[0.00033,-0.00033]] #08
#m_K, m_K_err = [0.493677,[0.000016,-0.000016]]
#m_pi, m_pi_err = [0.13957018,[0.00000035,-0.00000035]]
#m_tau, m_tau_err = [1.77684,[0.00017,-0.00017]]
#m_mu, m_mu_err = [0.1056583668,[0.0000000038,-0.0000000038]]
#m_u, m_u_err = [0.00240,[0.00090,-0.00090]]
#m_d, m_d_err = [0.00475,[0.00125,-0.00125]]
#m_s, m_s_err = [0.096,[0.030,-0.030]]
#m_c, m_c_err = [1.286,[0.042,-0.042]] #08
#m_b, m_b_err = [4.243,[0.043,-0.043]] #08
#
#Vub, Vub_err = [0.00350,[0.00015,-0.00014]]
#Vcd, Vcd_err = [0.224508,[0.00082,-0.00072]]
#Vcs, Vcs_err = [0.97347,[0.00019,-0.00019]]
#Vus, Vus_err = [0.22521,[0.00082,-0.00082]]
#Vud, Vud_err = [0.97430,[0.00019,-0.00019]]
#Vts, Vts_err = [0.04041,[0.00038,-0.00115]]
#Vtd, Vtd_err = [0.00859,[0.00027,-0.00029]]
#Vtb, Vtb_err = [0.999146,[0.000047,-0.000016]]
#Vcb, Vcb_err = [0.04117,[0.00038,-0.00115]]
#
#f_bplus, f_bplus_err = [0.1902,[0.0174,-0.0174]]
#f_dplus, f_dplus_err = [0.2077,[0.0056,-0.0056]]
#f_dsplus, f_dsplus_err = [0.2463,[0.0065,-0.0065]]
#f_Kpi, f_Kpi_err = [1.205,[0.0108,-0.0084]]
#
#delt_kpi, delt_kpi_err = [-0.0070,[0.0035,-0.0035]]
#delt_kpitau, delt_kpitau_err = [0.0003,[0,0]]
#
#rho, rho_err = [1.19,[0.057,-0.057]] #08
#delt_rd, delt_rd_err = [0.46,[0.01,-0.01]] #0802.3790
#
#tau_bplus, tau_bplus_err = [(1.638e-12)/hbar_gev,[(0.011e-12)/hbar_gev,-(0.011e-12)/hbar_gev]]
#tau_dplus, tau_dplus_err = [(1040e-15)/hbar_gev,[(7e-15)/hbar_gev,-(7e-15)/hbar_gev]]
#tau_dsplus, tau_dsplus_err = [(500e-15)/hbar_gev,[(7e-15)/hbar_gev,-(7e-15)/hbar_gev]]
#
#bplus_exp, bplus_err_exp = [1.4e-4,[0.4e-4,-0.4e-4]]
#dplus_exp, dplus_err_exp = [3.82e-4,[0.33e-4,-0.33e-4]]
#dsplus_exp, dsplus_err_exp = [6.6e-2,[0.6e-2,-0.6e-2]]
#kpi_exp, kpi_exp_err = [1.337,[0.003,-0.003]]
#kpitau_exp, kpitau_exp_err = [6.38e-2,[0.215e-2,-0.215e-2]]
#rd_exp, rd_exp_err = [0.416,[0.128,-0.128]] #08


# B+ -> tau+ nu
#mH_bplus, tanb_bplus, vub_loc = ckmel(Vub,Vub_err,m_u,m_u_err,m_b,m_b_err,m_bplus,m_bplus_err,m_tau,m_tau_err,f_bplus,f_bplus_err,tau_bplus,tau_bplus_err,bplus_exp,bplus_err_exp)
#mH_bplus, tanb_bplus = itera(m_bplus,m_bplus_err,m_mu,m_mu_err,Vub,Vub_err,f_bplus,f_bplus_err,tau_bplus,tau_bplus_err,m_u,m_u_err,m_b,m_b_err,bplusmu_exp,bplusmu_err_exp,1)
#plt.figure(figsize=(8,6))
#plt.scatter(tanb_bplus,mH_bplus,c='green')
#plt.ylabel('$\\log[m_{H+}$, GeV]',fontsize=18)
#plt.xlabel('$\\log[\\tan(\\beta)]$',fontsize=18)
#plt.title('$B^+\\to\\tau^+\\nu,\; V_{ub}$',fontsize=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.axis([-1,2,0,3.5])
#plt.show()
#plt.savefig('vbtau.png')

# D+ -> mu+ nu
#mH_dplus, tanb_dplus, vud_loc = ckmel(Vcd,Vcd_err,m_c,m_c_err,m_d,m_d_err,m_dplus,m_dplus_err,m_mu,m_mu_err,f_dplus,f_dplus_err,tau_dplus,tau_dplus_err,dplus_exp,dplus_err_exp)
#mH_dplus, tanb_dplus = itera(m_dplus,m_dplus_err,m_mu,m_mu_err,Vcd,Vcd_err,f_dplus,f_dplus_err,tau_dplus,tau_dplus_err,m_c,m_c_err,m_d,m_d_err,dplus_exp,dplus_err_exp,0.982*0.99)
#plt.figure(figsize=(8,6))
#plt.scatter(tanb_dplus,mH_dplus,c='green')
#plt.ylabel('$\\log[m_{H+}$, GeV]',fontsize=18)
#plt.xlabel('$\\log[\\tan(\\beta)]$',fontsize=18)
#plt.title('$D^+\\to\\mu^+\\nu,\; V_{cd}$',fontsize=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.axis([-1,2,0,3.5])
#plt.show()
#plt.savefig('vdmu.png')
#
## Ds+ -> tau+ nu
#mH_dsplus, tanb_dsplus, vuds_loc = ckmel(Vcs,Vcs_err,m_c,m_c_err,m_s,m_s_err,m_dsplus,m_dsplus_err,m_tau,m_tau_err,f_dsplus,f_dsplus_err,tau_dsplus,tau_dsplus_err,dsplus_exp,dsplus_err_exp)
#mH_dsplus, tanb_dsplus = itera(m_dsplus,m_dsplus_err,m_mu,m_mu_err,Vcs,Vcs_err,f_dsplus,f_dsplus_err,tau_dsplus,tau_dsplus_err,m_c,m_c_err,m_s,m_s_err,dsmu_exp,dsmu_err_exp,1)
#plt.figure(figsize=(8,6))
#plt.scatter(tanb_dsplus,mH_dsplus,c='green')
#plt.ylabel('$\\log[m_{H+}$, GeV]',fontsize=18)
#plt.xlabel('$\\log[\\tan(\\beta)]$',fontsize=18)
#plt.title('$D_s^+\\to\\tau^+\\nu,\; V_{cs}$',fontsize=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.axis([-1,2,0,3.5])
#plt.show()
##plt.savefig('vdstau.png')
#quit()
## (K -> mu)/(pi -> mu) + (tau -> K)/(tau -> pi)
#mH2, tanb2 = itera_kpi(m_K,m_K_err,m_pi,m_pi_err,m_mu,m_mu_err,m_tau,m_tau_err,Vus,Vus_err,Vud,Vud_err,f_Kpi,f_Kpi_err,delt_kpi,delt_kpi_err,delt_kpitau,delt_kpitau_err,m_s,m_s_err,m_d,m_d_err,m_u,m_u_err,kpi_exp,kpi_exp_err,kpitau_exp,kpi_exp_err)
##
#plt.figure(figsize=(8,6))
#plt.scatter(tanb2,mH2,c='green')
#plt.ylabel('$\\log[m_{H+}$, GeV]',fontsize=18)
#plt.xlabel('$\\log[\\tan(\\beta)]$',fontsize=18)
#plt.title('$K\\to\\mu\\nu/\\pi\\to\\mu\\nu$ & $\\tau\\to K\\nu/\\tau\\to\\pi\\nu$',fontsize=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.axis([-1,2,0,3.5])
##plt.show()
#plt.savefig('kpi.png')

# R(D) and ? R(D*)
#hrd,trd,chi_rds,chi_r = itera_rd(m_c,m_c_err,m_b,m_b_err,m_Brd,m_Brd_err,m_dplus,m_dplus_err,rho,rho_err,delt_rd,delt_rd_err,rd_exp,rd_exp_err)
#plt.figure(figsize=(8,6))
#plt.scatter(trd,hrd,c='cadetblue',marker=',')
#plt.ylabel('$\\log[m_{H+}$, GeV]',fontsize=18)
#plt.xlabel('$\\log[\\tan(\\beta)]$',fontsize=18)
#plt.title('$\\mathcal{R}(D)$ - 2020 Inputs',fontsize=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.axis([-1,2,0,3.5])
####plt.show()
#plt.savefig('rd.png')

#hrdst, trdst, chi_rdsst, chi_stmin = itera_rds(m_Brd,m_Brd_err,m_dstar,m_dstar_err,rhod,rhod_err,r01,r01_err,r11,r11_err,r21,r21_err,Vcb,Vcb_err,m_mu,m_mu_err,m_tau,m_tau_err,vev,vev_err,m_c,m_c_err,m_b,m_b_err,rdst_exp,rdst_exp_err)
# hrdn, trdn, chi_rdn, chi_nmin = itera_rdn(m_Brd,m_Brd_err,m_dplus,m_dplus_err,rhod,rhod_err,delta,delta_err,Vcb,Vcb_err,m_mu,m_mu_err,m_tau,m_tau_err,vev,vev_err,m_c,m_c_err,m_b,m_b_err,rd_exp,rd_exp_err)
#hrdn, trdn, chi_rdn, chi_nmin, hrdst, trdst, chi_rdst, chi_stmin = itera_rda(m_Brd,m_Brd_err,m_dplus,m_dplus_err,m_dstar,m_dstar_err,rhod,rhod_err,r01,r01_err,r11,r11_err,r21,r21_err,delta,delta_err,Vcb,Vcb_err,m_mu,m_mu_err,m_tau,m_tau_err,vev,vev_err,m_c,m_c_err,m_b,m_b_err,rd_exp,rd_exp_err,rdst_exp,rdst_exp_err)
#
#m2,m1=5.99,2.30
##hchi_rd, tchi_rd, rd_edges_e = chi_del(chi_nmin,chi_rdn,hrdn,trdn,m2)
#hchi_rd2, tchi_rd2, rd_edges = chi_del(chi_nmin,chi_rdn,hrdn,trdn,m1)
##hchi_rds, tchi_rds, rd_edges_e = chi_del(chi_nmin,chi_rdn,hrdn,trdn,m2)
#hchi_rds2, tchi_rds2, rds_edges = chi_del(chi_stmin,chi_rdst,hrdst,trdst,m1)

#plt.figure(figsize=(8,6))
#plt.scatter(trdst,hrdst,c='turquoise',marker=',')
#plt.scatter(trdn,hrdn,c='cadetblue',marker=',')
#for i, j in rd_edges[0]:
#    plt.plot(rd_edges[1][[i,j],0],rd_edges[1][[i,j],1],c='black',linestyle='--')
#for i, j in rds_edges[0]:
#    plt.plot(rds_edges[1][[i,j],0],rds_edges[1][[i,j],1],c='black',linestyle='--')
#plt.ylabel('$\\log[m_{H+}$, GeV]',fontsize=18)
#plt.xlabel('$\\log[\\tan(\\beta)]$',fontsize=18)
#plt.title('$\\mathcal{R}(D^{(*)}), 1\\sigma$',fontsize=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.axis([-1,2,0,3.5])
#plt.savefig('rd_both196sig.png')
#plt.show()
#os.system('play gumdrops.mp3')
#quit()

#hlepi,tlepi = itera_lepis(bplus_exp,bplus_err_exp,dplus_exp,dplus_err_exp,dsplus_exp,dsplus_err_exp,kpi_exp,kpi_exp_err,kpitau_exp,kpitau_exp_err,m_bplus,m_bplus_err,m_dplus,m_dplus_err,m_dsplus,m_dsplus_err,m_K,m_K_err,m_pi,m_pi_err,m_tau,m_tau_err,m_mu,m_mu_err,f_bplus,f_bplus_err,f_dplus,f_dplus_err,f_dsplus,f_dsplus_err,f_Kpi,f_Kpi_err,delt_kpi,delt_kpi_err,delt_kpitau,delt_kpitau_err,tau_bplus,tau_bplus_err,tau_dplus,tau_dplus_err,tau_dsplus,tau_dsplus_err,m_u,m_u_err,m_d,m_d_err,m_c,m_c_err,m_s,m_s_err,m_b,m_b_err,Vud,Vud_err,Vus,Vus_err,Vub,Vub_err,Vcd,Vcd_err,Vcs,Vcs_err,)
#plt.figure(figsize=(8,6))
#plt.scatter(tlepi,hlepi,c='green')
#plt.ylabel('$\\log[m_{H+}$, GeV]',fontsize=18)
#plt.xlabel('$\\log[\\tan(\\beta)]$',fontsize=18)
#plt.title('$M\\to l\\nu + \\tau \\to M\\nu$',fontsize=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.axis([-1,2,0,3.5])
##plt.show()
#plt.savefig('leps.png')
#quit()

###############   MIXING

mt, mt_err = [172.9,[0.4,-0.4]]
mW, mW_err = [80.379,[0.012,-0.012]]
mBd, mBd_err = [5.27964,[0.00013,-0.00013]]
mBs, mBs_err = [5.36688,[0.00017,-0.00017]]
lambda_QCD, QCD_err = [0.2275,[0.01433,-0.01372]]

etaB, etaB_err = [0.537856,[0,0]]

fBs, fBs_err = [0.0452,[0.0014,-0.0014]]
fBd, fBd_err = [0.0305,[0.0011,-0.0011]]
#fBs, fBs_err = [0.2303,[0.0013,-0.0013]]
#fBd, fBd_err = [0.190,[0.0013,-0.0013]]

#BBs, BBs_err = [1.290,[0.035,-0.035]]
#BBd, BBd_err = [1.268,[0.042,-0.042]]
BBs, BBs_err = [0.849,[0.023,-0.023]]
BBd, BBd_err = [0.835,[0.028,-0.028]]

delt_md, delt_md_err = [0.5064e12,[0.0019e12,-0.0019e12]]
delt_ms, delt_ms_err = [17.757e12,[0.021e12,-0.021e12]]
delt_md_expect, delt_md_err_exp = [0.533e12,[0.022e12,-0.036e12]]
delt_ms_expect, delt_ms_err_exp = [18.4e12,[0.7e12,-1.2e12]]

bdmix, bderr = mixing(mt,1,mW,1,Vtd,Vtb,etaB,mBd,fBd,BBd,1,lambda_QCD,m_b),error_mixing(mt,mt_err,1,mW,mW_err,1,Vtd,Vtd_err,Vtb,Vtb_err,etaB,etaB_err,mBd,mBd_err,fBd,fBd_err,BBd,BBd_err,1,[1,1],lambda_QCD,QCD_err,m_b,m_b_err)
bsmix, bserr = mixing(mt,1,mW,1,Vts,Vtb,etaB,mBs,fBs,BBs,1,lambda_QCD,m_b),error_mixing(mt,mt_err,1,mW,mW_err,1,Vts,Vts_err,Vtb,Vtb_err,etaB,etaB_err,mBs,mBs_err,fBs,fBs_err,BBs,BBs_err,1,[1,1],lambda_QCD,QCD_err,m_b,m_b_err)
print("Bd mixing =",bdmix*1e-12,"+",bderr[0]*1e-12,"-",bderr[1]*1e-12)
print("Bs mixing =",bsmix*1e-12,"+",bserr[0]*1e-12,"-",bserr[1]*1e-12)
quit()

# OLD INPUTS
#mt, mt_err = [172.4e3,[1.2e3,-1.2e3]]
#mW, mW_err = [80.398e3,[0.025e3,-0.025e3]]
#mBd, mBd_err = [5279.53,[0.33,-0.33]]
#mBs, mBs_err = [5366.3,[0.6,-0.6]]
#
#etaB, etaB_err = [0.5510,[0.0022,-0.0022]]
#
#fBs, fBs_err = [228.0,[17.3,-17.3]]
#fBd, fBd_err = [190,[15,-15]]
#
#BBs, BBs_err = [1.28,[0.036,-0.036]]
#BBd, BBd_err = [1.219,[0.05,-0.05]]
#
#delt_md, delt_md_err = [0.507e12,[0.005e12,-0.005e12]]
#delt_ms, delt_ms_err = [17.77e12,[0.12e12,-0.12e12]]
#delt_md_expect, delt_md_err_exp = [0.52e12,[0.02e12,-0.02e12]]
#delt_ms_expect, delt_ms_err_exp = [19.0e12,[6.6e12,-6.6e12]]

# B0d mixing
mH_md, tanb_md = itera_mix(mt,mt_err,mW,mW_err,Vtd,Vtd_err,Vtb,Vtb_err,etaB,etaB_err,mBd,mBd_err,fBd,fBd_err,BBd,BBd_err,delt_md,delt_md_err,delt_md_expect,delt_md_err_exp,lambda_QCD,QCD_err,m_b,m_b_err)
#x,y,z = error_mixing(mt,mt_err,1,mW,mW_err,1,Vtd,Vtd_err,Vtb,Vtb_err,etaB,etaB_err,mBd,mBd_err,fBd,fBd_err,BBd,BBd_err,delt_md,delt_md_err)
##print (1e-12)*(z+x)
##print (1e-12)*(z-y)
##print 1e-12*z
##
plt.figure(figsize=(8,6))
plt.scatter(tanb_md,mH_md,c='cornflowerblue')
plt.ylabel('$\\log[m_{H+}$, GeV]',fontsize=18)
plt.xlabel('$\\log[\\tan(\\beta)]$',fontsize=18)
plt.title('$B^0_q-\\bar{B}^0_q$',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.axis([-1,2,0,3.5])
plt.show()
#plt.savefig('bmix.png')

# B0s mixing
#mH_ms, tanb_ms = itera_mix(mt,mt_err,mW,mW_err,Vts,Vts_err,Vtb,Vtb_err,etaB,etaB_err,mBs,mBs_err,fBs,fBs_err,BBs,BBs_err,delt_ms,delt_ms_err,delt_ms_expect,delt_ms_err_exp)
#x,y,z = error_mixing(mt,mt_err,1,mW,mW_err,1,Vts,Vts_err,Vtb,Vtb_err,etaB,etaB_err,mBs,mBs_err,fBs,fBs_err,BBs,BBs_err,delt_md,delt_md_err)
#print (1e-12)*(z+x)
#print (1e-12)*(z-y)
#print 1e-12*z
#
##plt.figure()
#plt.scatter(tanb_ms,mH_ms,marker=',',c='cyan')
#plt.axis([-1,2,0,3])
#plt.ylabel('$\\log[m_{H+}$, GeV]')
#plt.xlabel('$\\log[\\tan(\\beta)]$')
#plt.title('$B^0_s-\\bar{B}^0_s$')
#plt.show()


###################### b to s gamma

mt1, mt1_err = [173.1,[0.9,-0.9]]
mW1, mW1_err = [80.379,[0.012,-0.012]]
mub = 1.095
hi = [626126/272277,-56281/51730,-3/7,-1/14,-0.6494,-0.038,-0.0185,-0.0057]
a = [14/23,16/23,6/23,-12/23,0.4086,-0.4223,-0.8994,0.1456]
A0,ac,at,a_s = [3.155e-2,2.8,-1.06e-4,36.2]
B0,bc,bt,b_s = [7.564e-1,-2.43e-1,-7.68e-4,-4.62]
delt_mc, delt_mt, delt_as = [0.04,1.8,0.002]
#C, C_err = [0.546,[0.033,-0.033]]
branch_c, branchc_err = [0.1065,[0.0016,-0.0016]]
branchs, branchs_err = [3.32e-4,[0.15e-4,-0.15e-4]]
gamc, gamc_err = [10.18e-2,[0.24e-2,-0.24e-2]]
gamu, gamu_err = [8.41e-4,[0.59e-4,-0.59e-4]]


# OLD INPUTS
#lambda_QCD, QCD_err = [0.22119,[0.02599,-0.02404]]
#mt1, mt1_err = [172.4,[1.2,-1.2]]
#mW1, mW1_err = [80.398,[0.025,-0.025]]
#mub = 8.2
#hi = [626126/272277,-56281/51730,-3/7,-1/14,-0.6494,-0.038,-0.0185,-0.0057]
#a = [14/23,16/23,6/23,-12/23,0.4086,-0.4223,-0.8994,0.1456]
#A0,ac,at,a_s = [3.155e-2,2.8,-1.06e-4,36.2]
#B0,bc,bt,b_s = [7.564e-1,-2.43e-1,-7.68e-4,-4.62]
#delt_mc, delt_mt, delt_as = [0.04,1.8,0.004]
###C, C_err = [0.546,[0.033,-0.033]]
#branch_c, branchc_err = [0.1052,[0.011,-0.011]]
#branchs, branchs_err = [3.52e-4,[0.25e-4,-0.25e-4]]
#gamc, gamc_err = [10.52e-2,[1.1e-2,-1.1e-2]]
#gamu, gamu_err = [1.4e-3,[0.58e-4,-0.58e-4]]

#bgam, gerr = bsgamma(mt1,mW1,mub,lambda_QCD,hi,a,1,1,A0,ac,at,a_s,B0,bc,bt,b_s,delt_mc,delt_mt,delt_as,branch_c,gamu,Vub,Vts,Vtb,Vcb,1/137), error_gamma(mt1,mt1_err,mW1,mW1_err,mub,lambda_QCD,QCD_err,hi,a,1,1,A0,ac,at,a_s,B0,bc,bt,b_s,delt_mc,delt_mt,delt_as,branch_c,branchc_err,gamu,gamu_err,Vub,Vub_err,Vts,Vts_err,Vtb,Vtb_err,Vcb,Vcb_err,1/137)
#upg = bgam*branch_c*np.sqrt((branchc_err[0]/branch_c)**2 + (gerr[0]/bgam)**2)
#lg = bgam*branch_c*np.sqrt((branchc_err[1]/branch_c)**2 + (gerr[1]/bgam)**2)
#print("b -> s gamma =",bgam*branch_c*1e4,"+",upg*1e4,"-",lg*1e4)
#quit()

#mH_gam, tanb_gam = iter_gamma(mt1,mt1_err,mW1,mW1_err,mub,lambda_QCD,QCD_err,hi,a,A0,ac,at,a_s,B0,bc,bt,b_s,delt_mc,delt_mt,delt_as,branch_c,branchc_err,gamu,gamu_err,Vub,Vub_err,Vts,Vts_err,Vtb,Vtb_err,Vcb,Vcb_err,1/137,branch_c,branchc_err,branchs,branchs_err)
##
#plt.figure(figsize=(8,6))
#plt.scatter(tanb_gam,mH_gam,marker=',',c='coral')
#plt.ylabel('$\\log[m_{H+}$, GeV]',fontsize=18)
#plt.xlabel('$\\log[\\tan(\\beta)]$',fontsize=18)
#plt.title('$b \\to s\\gamma$',fontsize=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.axis([-1,2,0,3.5])
#plt.show()
#plt.savefig('bsgamma.png')
#quit()

###################### b(s/d) to mumu

taubd, taubd_err = [(1.519e-12)/hbar_gev,[(0.004e-12)/hbar_gev,(-0.004e-12)/hbar_gev]] #ps
taubs, taubs_err = [(1.510e-12)/hbar_gev,[(0.004e-12)/hbar_gev,(-0.004e-12)/hbar_gev]]
mbd, mbd_err = [5.27964,[0.00013,-0.00013]] #GeV
mbs, mbs_err = [5.36688,[0.00017,-0.00017]]
bs_exp, bs_exp_err = [3.1e-9,[0.6e-9,-0.6e-9]] #hflav
bd_exp, bd_exp_err = [1.4e-10,[1.6e-10,-1.4e-10]] #pdg
sm, sm_err = [3.1e-9,[0.7e-9,-0.7e-9]]
wangle, wangle_err = [0.23155,[0.00004,-0.00004]]
higgs, higgs_err = [125.10,[0.14,-0.14]]
#mH_bmumu, tanb_bmumu = itera_bmumu(mt1,mt1_err,taubs,taubs_err,fBs,fBs_err,Vtb,Vtb_err,Vts,Vts_err,m_mu,m_mu_err,mbs,mbs_err,mW1,mW1_err,m_b,m_b_err,m_s,m_s_err,m_c,m_c_err,m_u,m_u_err,wangle,wangle_err,higgs,higgs_err,vev,vev_err,Vus,Vus_err,Vub,Vub_err,Vcs,Vcs_err,Vcb,Vcb_err,bs_exp,bs_exp_err)
###
#plt.figure(figsize=(8,6))
#plt.scatter(tanb_bmumu,mH_bmumu,marker=',',c='red')
#plt.axis([-1,2,0,3.5])
#plt.ylabel('$\\log[m_{H+}$, GeV]',fontsize=18)
#plt.xlabel('$\\log[\\tan(\\beta)]$',fontsize=18)
#plt.title('$M = 750\,GeV,\; m_{H^0} = 2\,TeV$\n$\\beta-\\alpha=\\pi/2+0.05$',fontsize=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.savefig('bcba9.png')
#plt.show()
##os.system('play gumdrops.mp3')
#quit()

###################### GLOBAL CONSTRAINT

#hl,tl,hb,tb,hg,tg,ha,ta = itera_firstglobal(bplus_exp,bplus_err_exp,dplus_exp,dplus_err_exp,dsplus_exp,dsplus_err_exp,delt_md,delt_md_err,delt_md_expect,delt_md_err_exp,kpi_exp,kpi_exp_err,kpitau_exp,kpitau_exp_err,branchs,branchs_err,branch_c,branchc_err,m_bplus,m_bplus_err,m_dplus,m_dplus_err,m_dsplus,m_dsplus_err,m_K,m_K_err,m_pi,m_pi_err,mBd,mBd_err,m_tau,m_tau_err,m_mu,m_mu_err,etaB,etaB_err,fBd,fBd_err,BBd,BBd_err,f_bplus,f_bplus_err,f_dplus,f_dplus_err,f_dsplus,f_dsplus_err,f_Kpi,f_Kpi_err,delt_kpi,delt_kpi_err,delt_kpitau,delt_kpitau_err,tau_bplus,tau_bplus_err,tau_dplus,tau_dplus_err,tau_dsplus,tau_dsplus_err,m_u,m_u_err,m_d,m_d_err,m_c,m_c_err,m_s,m_s_err,m_b,m_b_err,mt1,mt1_err,mt,mt_err,mW1,mW1_err,mW,mW_err,mub,lambda_QCD,QCD_err,hi,a,A0,ac,at,a_s,B0,bc,bt,b_s,delt_mc,delt_mt,delt_as,gamu,gamu_err,1/137,Vud,Vud_err,Vus,Vus_err,Vub,Vub_err,Vcd,Vcd_err,Vcs,Vcs_err,Vcb,Vcb_err,Vtd,Vtd_err,Vts,Vts_err,Vtb,Vtb_err)
hl,tl,hb,tb,hg,tg,ha,ta,hmu,tmu,hrd,trd,hl2,tl2,chi_ls,chi_ms,chi_gs,chi_as,chi_mus,chi_rds,chi_2s,chi_l,chi_m,chi_g,chi_a,chi_mu,chi_r,chi_2 = itera_global(bplus_exp,bplus_err_exp,dplus_exp,dplus_err_exp,dsplus_exp,dsplus_err_exp,delt_md,delt_md_err,delt_md_expect,delt_md_err_exp,kpi_exp,kpi_exp_err,kpitau_exp,kpitau_exp_err,branchs,branchs_err,branch_c,branchc_err,m_bplus,m_bplus_err,m_dplus,m_dplus_err,m_dsplus,m_dsplus_err,m_K,m_K_err,m_pi,m_pi_err,mBd,mBd_err,m_tau,m_tau_err,m_mu,m_mu_err,etaB,etaB_err,fBd,fBd_err,BBd,BBd_err,f_bplus,f_bplus_err,f_dplus,f_dplus_err,f_dsplus,f_dsplus_err,f_Kpi,f_Kpi_err,delt_kpi,delt_kpi_err,delt_kpitau,delt_kpitau_err,tau_bplus,tau_bplus_err,tau_dplus,tau_dplus_err,tau_dsplus,tau_dsplus_err,m_u,m_u_err,m_d,m_d_err,m_c,m_c_err,m_s,m_s_err,m_b,m_b_err,mt1,mt1_err,mt,mt_err,mW1,mW1_err,mW,mW_err,mub,lambda_QCD,QCD_err,hi,a,A0,ac,at,a_s,B0,bc,bt,b_s,delt_mc,delt_mt,delt_as,gamu,gamu_err,1/137,Vud,Vud_err,Vus,Vus_err,Vub,Vub_err,Vcd,Vcd_err,Vcs,Vcs_err,Vcb,Vcb_err,Vtd,Vtd_err,Vts,Vts_err,Vtb,Vtb_err,taubd,taubd_err,taubs,taubs_err,fBs,fBs_err,mbd,mbd_err,mbs,mbs_err,bs_exp,bs_exp_err,bd_exp,bd_exp_err,mBs,mBs_err,BBs,BBs_err,delt_ms,delt_ms_err,delt_ms_expect,delt_ms_err_exp,m_Brd,m_Brd_err,rho,rho_err,delt_rd,delt_rd_err,rd_exp,rd_exp_err,delta_b,delta_d,wangle,wangle_err,higgs,higgs_err,vev,vev_err,bplusmu_exp,bplusmu_err_exp,dsmu_exp,dsmu_err_exp)

if len(ha) > 2:
    print 10**min(ha)
#X = np.vstack((ta,ha)).T
#matr = cov_mat(X.T)
#print matr
#
#print 10**min(hl2)
#Y = np.vstack((tl2,hl2)).T
#matr2 = cov_mat(Y.T)
#print matr2
m1,m2 = 2.30,5.99

hchi_leps, tchi_leps, lep_edges_e = chi_del(chi_l,chi_ls,hl,tl,m2)
hchi_leps2, tchi_leps2, lep_edges = chi_del(chi_l,chi_ls,hl,tl,m1)

hchi_mix, tchi_mix, mix_edges_e = chi_del(chi_m,chi_ms,hb,tb,m2)
hchi_mix2, tchi_mix2, mix_edges = chi_del(chi_m,chi_ms,hb,tb,m1)

hchi_gam, tchi_gam, gam_edges_e = chi_del(chi_g,chi_gs,hg,tg,m2)
hchi_gam2, tchi_gam2, gam_edges = chi_del(chi_g,chi_gs,hg,tg,m1)

hchi_mu, tchi_mu, mu_edges_e = chi_del(chi_mu,chi_mus,hmu,tmu,m2)
hchi_mu2, tchi_mu2, mu_edges = chi_del(chi_mu,chi_mus,hmu,tmu,m1)

#hchi_a, tchi_a, a_edges_e = chi_del(chi_a[0],chi_as,ha,ta,m2)
#hchi_a2, tchi_a2, a_edges = chi_del(chi_a[0],chi_as,ha,ta,m1)

#print 10**min(hchi_a), 10**min(hchi_a2)
#print 10**min(tchi_a2)

#plt.figure(figsize=(8,6))
##plt.scatter(tb,hb,c='cornflowerblue')
#plt.scatter(tchi_mix,hchi_mix,c='cornflowerblue')
#for i, j in mix_edges_e[0]:
#    plt.plot(mix_edges_e[1][[i,j],0],mix_edges_e[1][[i,j],1],c='midnightblue',linestyle='--')
##plt.scatter(tl,hl,c='green')
#plt.scatter(tchi_leps,hchi_leps,c='green')
#for i, j in lep_edges_e[0]:
#    plt.plot(lep_edges_e[1][[i,j],0],lep_edges_e[1][[i,j],1],c='darkgreen',linestyle='--')
##plt.scatter(tg,hg,c='coral')
#plt.scatter(tchi_gam,hchi_gam,c='coral')
##plt.scatter(ta,ha,c='orange')
#plt.scatter(tchi_a,hchi_a,c='orange')
#for i, j in a_edges[0]:
#    plt.plot(a_edges[1][[i,j],0],a_edges[1][[i,j],1],c='chocolate',linestyle='--')
#plt.axis([-1,2,1,3])
#plt.ylabel('$\\log[m_{H+}$, GeV]',fontsize=18)
#plt.xlabel('$\\log[\\tan(\\beta)]$',fontsize=18)
#plt.title('Global Fit',fontsize=18)
#plt.annotate('$M\\to l\\nu+\\tau\\to M\\nu$',xy=(0.05,0.5),xycoords='axes fraction',fontsize=18)
#plt.annotate('$b\\to s\\gamma$',xy=(0.20,0.9),xycoords='axes fraction',fontsize=18)
#plt.annotate('All',xy=(0.55,0.9),xycoords='axes fraction',fontsize=18)
###plt.annotate('Global $\\to$',xy=(0.72,0.92),xycoords='axes fraction',fontsize=18)
#plt.annotate('$\\Delta M_q$',xy=(0.8,0.35),xycoords='axes fraction',fontsize=18)
###plt.annotate('$B_q \\to \\mu^+\\mu^-$',xy=(0.55,0.24),xycoords='axes fraction',fontsize=18,rotation=75)
###plt.annotate('$\\mathcal{R}(D)$',xy=(0.3,0.2),xycoords='axes fraction',fontsize=18,rotation=34)
##plt.show()
#plt.savefig('test.png')
#quit()

hchi_2, tchi_2, two_edges_e = chi_del(chi_2[0],chi_2s,hl2,tl2,m2)
hchi_22, tchi_22, two_edges = chi_del(chi_2[0],chi_2s,hl2,tl2,m1)

#print 10**min(hchi_a), 10**min(hchi_a2)
#print 10**min(tchi_a), 10**min(tchi_a2)
#print chi_a[0]/6
#print chi_a
print 10**min(hchi_2), 10**min(hchi_22)
print 10**min(tchi_22)
print chi_2[0]/9
print chi_2

plt.figure(figsize=(8,6))
plt.scatter(tchi_mix,hchi_mix,c='cornflowerblue')#,alpha=0.3)
for i, j in mix_edges_e[0]:
    plt.plot(mix_edges_e[1][[i,j],0],mix_edges_e[1][[i,j],1],c='midnightblue',linestyle='--')
plt.scatter(tchi_leps,hchi_leps,c='green')#,alpha=0.3)
for i, j in lep_edges_e[0]:
    plt.plot(lep_edges_e[1][[i,j],0],lep_edges_e[1][[i,j],1],c='darkgreen',linestyle='--')
plt.scatter(tchi_gam,hchi_gam,c='coral')#,alpha=0.3)
for i, j in gam_edges_e[0]:
    plt.plot(gam_edges_e[1][[i,j],0],gam_edges_e[1][[i,j],1],c='darkgoldenrod',linestyle='--')
plt.scatter(tchi_mu,hchi_mu,c='red')#,alpha=0.3)
for i, j in mu_edges_e[0]:
    plt.plot(mu_edges_e[1][[i,j],0],mu_edges_e[1][[i,j],1],c='chocolate',linestyle='--')
#plt.scatter(tchi_a,hchi_a,c='orange')#,alpha=0.3)
plt.scatter(tchi_2,hchi_2,c='darkorchid')#,alpha=0.3)
for i, j in two_edges[0]:
    plt.plot(two_edges[1][[i,j],0],two_edges[1][[i,j],1],c='plum',linestyle='--')
#plt.scatter(tchi_22,hchi_22,c='plum')#,alpha=0.3)
plt.axis([-1,2,0,3.5])
plt.ylabel('$\\log[m_{H+}$, GeV]',fontsize=18)
plt.xlabel('$\\log[\\tan(\\beta)]$',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.annotate('$M\\to l\\nu+\\tau\\to M\\nu$ \n $+ \\mathcal{R}(D)$',xy=(0.05,0.5),xycoords='axes fraction',fontsize=18)
plt.annotate('$\\Delta M_q$',xy=(0.75,0.35),xycoords='axes fraction',fontsize=18)
plt.annotate('$b\\to s\\gamma$',xy=(0.05,0.9),xycoords='axes fraction',fontsize=18)
plt.annotate('All',xy=(0.55,0.85),xycoords='axes fraction',fontsize=18)
plt.annotate('$B_s \\to \\mu^+\\mu^-$',xy=(0.5,0.65),xycoords='axes fraction',fontsize=18)#,rotation=75)
plt.title('$M = 750\,GeV,\; m_{H^0} = 2\,TeV,\; \\beta-\\alpha = \\frac{\\pi}{2}$',fontsize=18)
plt.savefig('globals1.png')
#plt.show()
os.system('play gumdrops.mp3')
