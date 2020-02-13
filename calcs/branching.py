from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from fitting import *

g_gev = (1.1663787e-5)**2
hbar_gev = 6.582119514e-25
g_mev = (1.1663787e-11)**2
hbar_mev = 6.582119514e-22

################ LEPTONIC AND SEMILEPTONIC DECAY

m_bplus, m_bplus_err = [5.27933,[0.00013,-0.00013]]
m_dplus08, m_dplus08_err = [1.86962,[0.00020,-0.00020]] #08
m_dplus, m_dplus_err = [1.86965,[0.00005,-0.00005]]
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

Vub, Vub_err = [0.003683,[0.000075,-0.000061]]
Vcd, Vcd_err = [0.224701,[0.000254,-0.000058]]
Vcs, Vcs_err = [0.973539,[0.000038,-0.000060]]
Vus, Vus_err = [0.224834,[0.000252,-0.000059]]
Vud, Vud_err = [0.974390,[0.000014,-0.000058]]

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
tau_dplus, tau_dplus_err = [(1040e-15)/hbar_gev,[(7e-15)/hbar_gev,-(7e-15)/hbar_gev]]
tau_dsplus, tau_dsplus_err = [(504e-15)/hbar_gev,[(4e-15)/hbar_gev,-(4e-15)/hbar_gev]]

bplus_exp, bplus_err_exp = [1.09e-4,[0.24e-4,-0.24e-4]]
dplus_exp, dplus_err_exp = [3.82e-4,[0.33e-4,-0.33e-4]]
dsplus_exp, dsplus_err_exp = [0.0548,[0.0023,-0.0023]]
kpi_exp, kpi_exp_err = [1.337,[0.0032,-0.0032]]
kpitau_exp, kpitau_exp_err = [6.438e-2,[9.384e-4,-9.384e-4]]
rd08_exp, rd08_exp_err = [0.416,[0.128,-0.128]] #08
rd_exp, rd_exp_err = [0.34,[0.03,-0.03]] #1909.12524

# B+ -> tau+ nu
#mH_bplus, tanb_bplus = itera(m_bplus,m_bplus_err,m_tau,m_tau_err,Vub,Vub_err,f_bplus,f_bplus_err,tau_bplus,tau_bplus_err,m_u,m_u_err,m_b,m_b_err,bplus_exp,bplus_err_exp)
#
#plt.figure(figsize=(8,6))
#plt.scatter(tanb_bplus,mH_bplus,c='green')
#plt.ylabel('$\\log[m_{H+}$, GeV]',fontsize=18)
#plt.xlabel('$\\log[\\tan(\\beta)]$',fontsize=18)
#plt.title('$B^+\\to\\tau^+\\nu$',fontsize=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.axis([-1,2,0,3.5])
##plt.show()
#plt.savefig('btau.png')

# D+ -> mu+ nu
#mH_dplus, tanb_dplus = itera(m_dplus,m_dplus_err,m_mu,m_mu_err,Vcd,Vcd_err,f_dplus,f_dplus_err,tau_dplus,tau_dplus_err,m_c,m_c_err,m_d,m_d_err,dplus_exp,dplus_err_exp)
#
#plt.figure(figsize=(8,6))
#plt.scatter(tanb_dplus,mH_dplus,c='green')
#plt.ylabel('$\\log[m_{H+}$, GeV]',fontsize=18)
#plt.xlabel('$\\log[\\tan(\\beta)]$',fontsize=18)
#plt.title('$D^+\\to\\mu^+\\nu$',fontsize=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.axis([-1,2,0,3.5])
##plt.show()
#plt.savefig('dmu.png')
#
## Ds+ -> tau+ nu
#mH_dsplus, tanb_dsplus = itera(m_dsplus,m_dsplus_err,m_tau,m_tau_err,Vcs,Vcs_err,f_dsplus,f_dsplus_err,tau_dsplus,tau_dsplus_err,m_c,m_c_err,m_s,m_s_err,dsplus_exp,dsplus_err_exp)
#
#plt.figure(figsize=(8,6))
#plt.scatter(tanb_dsplus,mH_dsplus,c='green')
#plt.ylabel('$\\log[m_{H+}$, GeV]',fontsize=18)
#plt.xlabel('$\\log[\\tan(\\beta)]$',fontsize=18)
#plt.title('$D_s^+\\to\\tau^+\\nu$',fontsize=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.axis([-1,2,0,3.5])
##plt.show()
#plt.savefig('dstau.png')
#
## (K -> mu)/(pi -> mu) + (tau -> K)/(tau -> pi)
#mH2, tanb2 = itera_kpi(m_K,m_K_err,m_pi,m_pi_err,m_mu,m_mu_err,m_tau,m_tau_err,Vus,Vus_err,Vud,Vud_err,f_Kpi,f_Kpi_err,delt_kpi,delt_kpi_err,delt_kpitau,delt_kpitau_err,m_s,m_s_err,m_d,m_d_err,m_u,m_u_err,kpi_exp,kpi_exp_err,kpitau_exp,kpi_exp_err)
#
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
hrd,trd,chi_rds,chi_r = itera_rd(m_c,m_c_err,m_b,m_b_err,m_Brd,m_Brd_err,m_dplus,m_dplus_err,rho,rho_err,delt_rd,delt_rd_err,rd_exp,rd_exp_err)
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
#hrd,trd = itera_rd(m_c08,m_c08_err,m_b08,m_b08_err,m_B08,m_B08_err,m_dplus08,m_dplus08_err,rho08,rho08_err,delt_rd,delt_rd_err,rd08_exp,rd08_exp_err)
#plt.figure(figsize=(8,6))
#plt.scatter(trd,hrd,c='cadetblue',marker=',')
#plt.ylabel('$\\log[m_{H+}$, GeV]',fontsize=18)
#plt.xlabel('$\\log[\\tan(\\beta)]$',fontsize=18)
#plt.title('$\\mathcal{R}(D)$ - 2008 Inputs',fontsize=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.axis([-1,2,0,3.5])
####plt.show()
#plt.savefig('rd_08.png')

#quit()

#hlepi,tlepi = itera_lepis(bplus_exp,bplus_err_exp,dplus_exp,dplus_err_exp,dsplus_exp,dsplus_err_exp,kpi_exp,kpi_exp_err,kpitau_exp,kpitau_exp_err,m_bplus,m_bplus_err,m_dplus,m_dplus_err,m_dsplus,m_dsplus_err,m_K,m_K_err,m_pi,m_pi_err,m_tau,m_tau_err,m_mu,m_mu_err,f_bplus,f_bplus_err,f_dplus,f_dplus_err,f_dsplus,f_dsplus_err,f_Kpi,f_Kpi_err,delt_kpi,delt_kpi_err,delt_kpitau,delt_kpitau_err,tau_bplus,tau_bplus_err,tau_dplus,tau_dplus_err,tau_dsplus,tau_dsplus_err,m_u,m_u_err,m_d,m_d_err,m_c,m_c_err,m_s,m_s_err,m_b,m_b_err,Vud,Vud_err,Vus,Vus_err,Vub,Vub_err,Vcd,Vcd_err,Vcs,Vcs_err,)
#plt.figure(figsize=(8,6))
#plt.scatter(tlepi,hlepi,c='green')
#plt.ylabel('$\\log[m_{H+}$, GeV]',fontsize=18)
#plt.xlabel('$\\log[\\tan(\\beta)]$',fontsize=18)
#plt.title('$M\\to l\\nu + \\tau \\to M\\nu',fontsize=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.axis([-1,2,0,3.5])
##plt.show()
#plt.savefig('leps.png')

###############   MIXING

mt, mt_err = [172.9e3,[0.4e3,-0.4e3]]
mW, mW_err = [80.379e3,[0.012e3,-0.012e3]]
mBd, mBd_err = [5279.64,[0.13,-0.13]]
mBs, mBs_err = [5366.88,[0.17,-0.17]]

Vts, Vts_err = [0.04090,[0.00026,-0.00076]]
Vtd, Vtd_err = [0.008545,[0.000075,-0.000157]]
Vtb, Vtb_err = [0.999127,[0.000032,-0.000012]]

etaB, etaB_err = [0.537856,[0,0]]

fBs, fBs_err = [230.3,[1.3,-1.3]]
fBd, fBd_err = [190,[1.3,-1.3]]

BBs, BBs_err = [1.290,[0.035,-0.035]]
BBd, BBd_err = [1.268,[0.042,-0.042]]

delt_md, delt_md_err = [0.5064e12,[0.0019e12,-0.0019e12]]
delt_ms, delt_ms_err = [17.757e12,[0.021e12,-0.021e12]]
delt_md_expect, delt_md_err_exp = [0.533e12,[0.022e12,-0.036e12]]
delt_ms_expect, delt_ms_err_exp = [18.4e12,[0.7e12,-1.2e12]]

# B0d mixing
#mH_md, tanb_md = itera_mix(mt,mt_err,mW,mW_err,Vtd,Vtd_err,Vtb,Vtb_err,etaB,etaB_err,mBd,mBd_err,fBd,fBd_err,BBd,BBd_err,delt_md,delt_md_err,delt_md_expect,delt_md_err_exp)
##x,y,z = error_mixing(mt,mt_err,1,mW,mW_err,1,Vtd,Vtd_err,Vtb,Vtb_err,etaB,etaB_err,mBd,mBd_err,fBd,fBd_err,BBd,BBd_err,delt_md,delt_md_err)
##print (1e-12)*(z+x)
##print (1e-12)*(z-y)
##print 1e-12*z
##
#plt.figure(figsize=(8,6))
#plt.scatter(tanb_md,mH_md,c='cornflowerblue')
#plt.ylabel('$\\log[m_{H+}$, GeV]',fontsize=18)
#plt.xlabel('$\\log[\\tan(\\beta)]$',fontsize=18)
#plt.title('$B^0_q-\\bar{B}^0_q$',fontsize=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.axis([-1,2,0,3.5])
##plt.show()
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

lambda_QCD, QCD_err = [0.224972,[0.012412,-0.012886]]
mt1, mt1_err = [173.1,[0.9,-0.9]]
mW1, mW1_err = [80.379,[0.012,-0.012]]
mub = 1.5
hi = [626126/272277,-56281/51730,-3/7,-1/14,-0.6494,-0.038,-0.0185,-0.0057]
a = [14/23,16/23,6/23,-12/23,0.4086,-0.4223,-0.8994,0.1456]
A0,ac,at,a_s = [3.155e-2,2.8,-1.06e-4,36.2]
B0,bc,bt,b_s = [7.564e-1,-2.43e-1,-7.68e-4,-4.62]
delt_mc, delt_mt, delt_as = [0.04,1.8,0.002]
#C, C_err = [0.546,[0.033,-0.033]]
Vcb, Vcb_err = [0.04162,[0.00026,-0.00080]]
branch_c, branchc_err = [0.1065,[0.0016,-0.0016]]
branchs, branchs_err = [3.32e-4,[0.15e-4,-0.15e-4]]
gamc, gamc_err = [10.18e-2,[0.24e-2,-0.24e-2]]
gamu, gamu_err = [8.41e-4,[0.59e-4,-0.59e-4]]

#mH_gam, tanb_gam = iter_gamma(mt1,mt1_err,mW1,mW1_err,mub,lambda_QCD,QCD_err,hi,a,A0,ac,at,a_s,B0,bc,bt,b_s,delt_mc,delt_mt,delt_as,branch_c,branchc_err,gamu,gamu_err,Vub,Vub_err,Vts,Vts_err,Vtb,Vtb_err,Vcb,Vcb_err,1/137,branch_c,branchc_err,branchs,branchs_err)
#
#plt.figure(figsize=(8,6))
#plt.scatter(tanb_gam,mH_gam,marker=',',c='coral')
#plt.ylabel('$\\log[m_{H+}$, GeV]',fontsize=18)
#plt.xlabel('$\\log[\\tan(\\beta)]$',fontsize=18)
#plt.title('$b \\to s\\gamma$',fontsize=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.axis([-1,2,0,3.5])
##plt.show()
#plt.savefig('bsgamma.png')
#quit()

###################### b(s/d) to mumu

taubd, taubd_err = [1.519,[0.004,-0.004]] #ps
taubs, taubs_err = [1.510,[0.004,-0.004]]
mbd, mbd_err = [5.27964,[0.00013,-0.00013]] #GeV
mbs, mbs_err = [5.36688,[0.00017,-0.00017]]
bs_exp, bs_exp_err = [3.1e-9,[0.6e-9,-0.6e-9]] #hflav
bd_exp, bd_exp_err = [1.4e-10,[1.6e-10,-1.4e-10]] #pdg
sm, sm_err = [3.1e-9,[0.7e-9,-0.7e-9]]
#mH_bmumu, tanb_bmumu = itera_bmumu(mt1,mt1_err,taubd,taubd_err,taubs,taubs_err,fBd,fBd_err,fBs,fBs_err,Vtd,Vtd_err,Vts,Vts_err,m_mu,m_mu_err,mbd,mbd_err,mbs,mbs_err,mW1,mW1_err,bs_exp,bs_exp_err,bd_exp,bd_exp_err)

##plt.figure()
#plt.scatter(tanb_bmumu,mH_bmumu,marker=',',c='red')
#plt.axis([-1,2,0,3])
#plt.ylabel('$\\log[m_{H+}$, GeV]')
#plt.xlabel('$\\log[\\tan(\\beta)]$')
#plt.show()
#plt.savefig('bmumu.png')

###################### GLOBAL CONSTRAINT

#hl,tl,hb,tb,hg,tg,ha,ta = itera_firstglobal(bplus_exp,bplus_err_exp,dplus_exp,dplus_err_exp,dsplus_exp,dsplus_err_exp,delt_md,delt_md_err,delt_md_expect,delt_md_err_exp,kpi_exp,kpi_exp_err,kpitau_exp,kpitau_exp_err,branchs,branchs_err,branch_c,branchc_err,m_bplus,m_bplus_err,m_dplus,m_dplus_err,m_dsplus,m_dsplus_err,m_K,m_K_err,m_pi,m_pi_err,mBd,mBd_err,m_tau,m_tau_err,m_mu,m_mu_err,etaB,etaB_err,fBd,fBd_err,BBd,BBd_err,f_bplus,f_bplus_err,f_dplus,f_dplus_err,f_dsplus,f_dsplus_err,f_Kpi,f_Kpi_err,delt_kpi,delt_kpi_err,delt_kpitau,delt_kpitau_err,tau_bplus,tau_bplus_err,tau_dplus,tau_dplus_err,tau_dsplus,tau_dsplus_err,m_u,m_u_err,m_d,m_d_err,m_c,m_c_err,m_s,m_s_err,m_b,m_b_err,mt1,mt1_err,mt,mt_err,mW1,mW1_err,mW,mW_err,mub,lambda_QCD,QCD_err,hi,a,A0,ac,at,a_s,B0,bc,bt,b_s,delt_mc,delt_mt,delt_as,gamu,gamu_err,1/137,Vud,Vud_err,Vus,Vus_err,Vub,Vub_err,Vcd,Vcd_err,Vcs,Vcs_err,Vcb,Vcb_err,Vtd,Vtd_err,Vts,Vts_err,Vtb,Vtb_err)
hl,tl,hb,tb,hg,tg,ha,ta,hmu,tmu,hrd,trd,hl2,tl2,chi_ls,chi_ms,chi_gs,chi_as,chi_mus,chi_rds,chi_2s,chi_l,chi_m,chi_g,chi_a,chi_mu,chi_r,chi_2 = itera_global(bplus_exp,bplus_err_exp,dplus_exp,dplus_err_exp,dsplus_exp,dsplus_err_exp,delt_md,delt_md_err,delt_md_expect,delt_md_err_exp,kpi_exp,kpi_exp_err,kpitau_exp,kpitau_exp_err,branchs,branchs_err,branch_c,branchc_err,m_bplus,m_bplus_err,m_dplus,m_dplus_err,m_dsplus,m_dsplus_err,m_K,m_K_err,m_pi,m_pi_err,mBd,mBd_err,m_tau,m_tau_err,m_mu,m_mu_err,etaB,etaB_err,fBd,fBd_err,BBd,BBd_err,f_bplus,f_bplus_err,f_dplus,f_dplus_err,f_dsplus,f_dsplus_err,f_Kpi,f_Kpi_err,delt_kpi,delt_kpi_err,delt_kpitau,delt_kpitau_err,tau_bplus,tau_bplus_err,tau_dplus,tau_dplus_err,tau_dsplus,tau_dsplus_err,m_u,m_u_err,m_d,m_d_err,m_c,m_c_err,m_s,m_s_err,m_b,m_b_err,mt1,mt1_err,mt,mt_err,mW1,mW1_err,mW,mW_err,mub,lambda_QCD,QCD_err,hi,a,A0,ac,at,a_s,B0,bc,bt,b_s,delt_mc,delt_mt,delt_as,gamu,gamu_err,1/137,Vud,Vud_err,Vus,Vus_err,Vub,Vub_err,Vcd,Vcd_err,Vcs,Vcs_err,Vcb,Vcb_err,Vtd,Vtd_err,Vts,Vts_err,Vtb,Vtb_err,taubd,taubd_err,taubs,taubs_err,fBs,fBs_err,mbd,mbd_err,mbs,mbs_err,bs_exp,bs_exp_err,bd_exp,bd_exp_err,mBs,mBs_err,BBs,BBs_err,delt_ms,delt_ms_err,delt_ms_expect,delt_ms_err_exp,m_Brd,m_Brd_err,rho,rho_err,delt_rd,delt_rd_err,rd_exp,rd_exp_err)

#print 10**min(ha)
#X = np.vstack((ta,ha)).T
#matr = cov_mat(X.T)
#print matr
#
#print 10**min(hl2)
#Y = np.vstack((tl2,hl2)).T
#matr2 = cov_mat(Y.T)
#print matr2

#plt.figure(figsize=(8,6))
#plt.scatter(tmu,hmu,c='red')
#plt.scatter(trd,hrd,c='cadetblue')
#plt.scatter(tl,hl,c='green')
#plt.scatter(tb,hb,c='cornflowerblue')
#plt.scatter(tg,hg,c='coral')
#plt.scatter(ta,ha,c='orange')
#plt.scatter(tl2,hl2,c='darkorchid')
#plt.axis([-1,2,0,3.5])
#plt.ylabel('$\\log[m_{H+}$, GeV]',fontsize=18)
#plt.xlabel('$\\log[\\tan(\\beta)]$',fontsize=18)
#plt.title('Global Fit',fontsize=18)
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.annotate('$M\\to l\\nu+\\tau\\to M\\nu$',xy=(0.05,0.5),xycoords='axes fraction',fontsize=18)
#plt.annotate('$b\\to s\\gamma$',xy=(0.20,0.85),xycoords='axes fraction',fontsize=18)
#plt.annotate('Minimal Global',xy=(0.48,0.85),xycoords='axes fraction',fontsize=18)
#plt.annotate('Global $\\to$',xy=(0.72,0.92),xycoords='axes fraction',fontsize=18)
#plt.annotate('$\\Delta M_q$',xy=(0.8,0.35),xycoords='axes fraction',fontsize=18)
#plt.annotate('$B_q \\to \\mu^+\\mu^-$',xy=(0.55,0.24),xycoords='axes fraction',fontsize=18,rotation=75)
#plt.annotate('$\\mathcal{R}(D)$',xy=(0.3,0.2),xycoords='axes fraction',fontsize=18,rotation=34)
#plt.show()
#plt.savefig('global_cl.png')

m1 = 4.61
hchi_leps, tchi_leps = chi_del(chi_l,chi_ls,hl,tl,m1)
hchi_mix, tchi_mix = chi_del(chi_m,chi_ms,hb,tb,m1)
hchi_gam, tchi_gam = chi_del(chi_g,chi_gs,hg,tg,m1)
hchi_mu, tchi_mu = chi_del(chi_mu,chi_mus,hmu,tmu,m1)
hchi_rd, tchi_rd = chi_del(chi_r,chi_rds,hrd,trd,m1)
hchi_a, tchi_a = chi_del(chi_a,chi_as,ha,ta,m1)
hchi_2, tchi_2 = chi_del(chi_2,chi_2s,hl2,tl2,m1)
#hchi_leps, tchi_leps = chi_del2(chi_l,chi_ls)
#hchi_mix, tchi_mix = chi_del2(chi_m,chi_ms)
#hchi_gam, tchi_gam = chi_del2(chi_g,chi_gs)
#hchi_mu, tchi_mu = chi_del2(chi_mu,chi_mus)
#hchi_rd, tchi_rd = chi_del2(chi_r,chi_rds)
#hchi_a, tchi_a = chi_del2(chi_a,chi_as)
#hchi_2, tchi_2 = chi_del2(chi_2,chi_2s)

print 10**min(hchi_a)
if len(hchi_2) > 0:
    print 10**min(hchi_2)

plt.figure(figsize=(8,6))
plt.scatter(tchi_mix,hchi_mix,c='cornflowerblue')#,alpha=0.3)
plt.scatter(tchi_mu,hchi_mu,c='red')#,alpha=0.3)
plt.scatter(tchi_rd,hchi_rd,c='cadetblue')#,alpha=0.3)
plt.scatter(tchi_leps,hchi_leps,c='green')#,alpha=0.3)
plt.scatter(tchi_gam,hchi_gam,c='coral')#,alpha=0.3)
plt.scatter(tchi_a,hchi_a,c='orange')#,alpha=0.3)
plt.scatter(tchi_2,hchi_2,c='darkorchid')#,alpha=0.3)
plt.axis([-1,2,0,3.5])
plt.ylabel('$\\log[m_{H+}$, GeV]',fontsize=18)
plt.xlabel('$\\log[\\tan(\\beta)]$',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.annotate('$M\\to l\\nu+\\tau\\to M\\nu$',xy=(0.05,0.5),xycoords='axes fraction',fontsize=18)
plt.annotate('$\\Delta M_q$',xy=(0.85,0.35),xycoords='axes fraction',fontsize=18)
plt.annotate('$\\mathcal{R}(D)$',xy=(0.3,0.2),xycoords='axes fraction',fontsize=18,rotation=34)
plt.annotate('$b\\to s\\gamma$',xy=(0.20,0.85),xycoords='axes fraction',fontsize=18)
plt.annotate('Minimal Global',xy=(0.48,0.85),xycoords='axes fraction',fontsize=18)
plt.annotate('Global $\\to$',xy=(0.72,0.92),xycoords='axes fraction',fontsize=18)
plt.annotate('$B_q \\to \\mu^+\\mu^-$',xy=(0.55,0.24),xycoords='axes fraction',fontsize=18,rotation=75)
plt.show()
plt.savefig('testing.png')
