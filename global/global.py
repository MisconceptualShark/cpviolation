from __future__ import division
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from functions import *
from fitting import *
from tdfs import *
import os

g_gev = (1.1663787e-5)**2
hbar_gev = 6.582119514e-25
g_mev = (1.1663787e-11)**2
hbar_mev = 6.582119514e-22

### Masses
# W, h, neutral B mesons
higgs, higgs_err = [125.10,[0.14,-0.14]]
mW, mW_err = [80.379,[0.012,-0.012]]
mZ, mZ_err = [91.1876,[0.0021,-0.0021]]
mBd, mBd_err = [5.27964,[0.00013,-0.00013]]
mBs, mBs_err = [5.36688,[0.00017,-0.00017]]

# Charged mesons, quarks, leptons
m_bplus, m_bplus_err = [5.27933,[0.00013,-0.00013]]
m_dplus, m_dplus_err = [1.86965,[0.00005,-0.00005]]
m_dstar, m_dstar_err = [2.1122,[0.0004,-0.0004]]
m_dsplus, m_dsplus_err = [1.96834,[0.00007,-0.00007]]
m_K, m_K_err = [0.493677,[0.000016,-0.000016]]
m_pi, m_pi_err = [0.13957061,[0.00000024,-0.00000024]]
m_tau, m_tau_err = [1.77686,[0.00012,-0.00012]]
m_mu, m_mu_err = [0.1056583745,[0.0000000024,-0.0000000024]]
m_u, m_u_err = [0.00216,[0.00049,-0.00026]]
m_d, m_d_err = [0.00467,[0.00048,-0.00017]]
m_s, m_s_err = [0.093,[0.011,-0.005]]
m_c, m_c_err = [1.27,[0.02,-0.02]]
m_b, m_b_err = [4.18,[0.03,-0.02]]
mt, mt_err = [172.9,[0.4,-0.4]]

### CKM
Vud, Vud_err = [0.974390,[0.000014,-0.000058]]
Vus, Vus_err = [0.224834,[0.000252,-0.000059]]
Vub, Vub_err = [0.003683,[0.000075,-0.000061]]
Vcd, Vcd_err = [0.224701,[0.000254,-0.000058]]
Vcs, Vcs_err = [0.973539,[0.000038,-0.000060]]
Vcb, Vcb_err = [0.04162,[0.00026,-0.00080]]
Vtd, Vtd_err = [0.008545,[0.000075,-0.000157]]
Vts, Vts_err = [0.04090,[0.00026,-0.00076]]
Vtb, Vtb_err = [0.999127,[0.000032,-0.000012]]

### Decay constants, Bags, etc
f_bplus, f_bplus_err = [0.190,[0.0013,-0.0013]] # fBd same as bplus

f_dplus, f_dplus_err = [0.212,[0.0007,-0.0007]]
f_dsplus, f_dsplus_err = [0.2499,[0.0005,-0.0005]]
f_Kpi, f_Kpi_err = [1.1932,[0.0019,-0.0019]]
fBs, fBs_err = [0.2303,[0.0013,-0.0013]]
f2Bs, f2Bs_err = [0.0452,[0.0014,-0.0014]] # f^2 * Bag(mu) (need etaB = 0.8...)
f2Bd, f2Bd_err = [0.0305,[0.0011,-0.0011]]

# deltas for K/pi to tau and tau to K/pi
delt_kpi, delt_kpi_err = [-0.0069,[0.0017,-0.0017]]
delt_kpitau, delt_kpitau_err = [0.0003,[0,0]]

# Radiative corrections
delta_b,delta_d = 0.982,0.99*0.982

#etaB, etaB_err = [0.537856,[0,0]]
etaB, etaB_err = [0.838606,[0,0]]

BBs, BBs_err = [1.290,[0.035,-0.035]] # Bag parameters with 1.519 factor (need etaB = 0.5...)
BBd, BBd_err = [1.268,[0.042,-0.042]]

### R(D) parameters
rho, rho_err = [1.131,[0.033,-0.033]] #rho for R(D)
rhod, rhod_err = [1.122,[0.024,-0.024]] #rho for R(D*)
delt_rd, delt_rd_err = [0.46,[0.01,-0.01]] #delta for parameterisation in 0907.5135
delta, delta_err = [1,[1,-1]] #delta for full integration in 1705.02456
r01, r01_err = [1.85,[0.05,-0.05]] #inputs for 1705.02456
r11, r11_err = [1.270,[0.026,-0.026]]
r21, r21_err = [0.852,[0.018,-0.018]]

### Lifetimes
tau_bplus, tau_bplus_err = [(1.638e-12)/hbar_gev,[(0.004e-12)/hbar_gev,-(0.004e-12)/hbar_gev]]
tau_dplus, tau_dplus_err = [(1040e-15)/hbar_gev,[(7e-15)/hbar_gev,-(7e-15)/hbar_gev]]
tau_dsplus, tau_dsplus_err = [(504e-15)/hbar_gev,[(4e-15)/hbar_gev,-(4e-15)/hbar_gev]]
taubd, taubd_err = [(1.519e-12)/hbar_gev,[(0.004e-12)/hbar_gev,(-0.004e-12)/hbar_gev]]
taubs, taubs_err = [(1.510e-12)/hbar_gev,[(0.004e-12)/hbar_gev,(-0.004e-12)/hbar_gev]]

### Experiment Values
bplus_exp, bplus_err_exp = [1.09e-4,[0.24e-4,-0.24e-4]] #to tau
dplus_exp, dplus_err_exp = [3.77e-4,[0.17e-4,-0.17e-4]] #to mu
dsplus_exp, dsplus_err_exp = [0.0548,[0.0023,-0.0023]] #to tau
bplusmu_exp, bplusmu_err_exp = [5.3e-7,[2.2e-7,-2.2e-7]] #to mu
dsmu_exp, dsmu_err_exp = [5.52e-3,[0.16e-3,-0.16e-3]] #to mu
kpi_exp, kpi_exp_err = [1.337,[0.0032,-0.0032]] #K/pi to tau
kpitau_exp, kpitau_exp_err = [6.438e-2,[9.384e-4,-9.384e-4]] #tau to K/pi
rd_exp, rd_exp_err = [0.34,[0.03,-0.03]] #R(D)
rdst_exp, rdst_exp_err = [0.295,[0.014,-0.014]] #R(D*)
delt_md, delt_md_err = [0.5064e12,[0.0019e12,-0.0019e12]] #Bd mix exp
delt_ms, delt_ms_err = [17.757e12,[0.021e12,-0.021e12]] #Bs mix exp
delt_md_SM, delt_md_err_SM = [0.533e12,[0.022e12,-0.036e12]] #Bd mix SM (Alex)
delt_ms_SM, delt_ms_err_SM = [18.4e12,[0.7e12,-1.2e12]] #Bs mix SM (Alex)
branch_c, branchc_err = [0.1065,[0.0016,-0.0016]] #B to Xc l nu
branchs, branchs_err = [3.32e-4,[0.15e-4,-0.15e-4]] #B to Xs gamma
gamu, gamu_err = [8.41e-4,[0.59e-4,-0.59e-4]] #B to Xu l nu
bs_exp, bs_exp_err = [3.1e-9,[0.7e-9,-0.7e-9]] #Bs to mu mu
bd_exp, bd_exp_err = [1.4e-10,[1.6e-10,-1.4e-10]] #Bd to mu mu
bssm, bssm_err = [3.57e-9,[0.17e-9,-0.17e-9]] #Bs to mu mu SM

### b sgamma stuff
mub = 1.095
hi = [626126/272277,-56281/51730,-3/7,-1/14,-0.6494,-0.038,-0.0185,-0.0057]
a = [14/23,16/23,6/23,-12/23,0.4086,-0.4223,-0.8994,0.1456]
A0,ac,at,a_s = [3.155e-2,2.8,-1.06e-4,36.2]
B0,bc,bt,b_s = [7.564e-1,-2.43e-1,-7.68e-4,-4.62]
delt_mc, delt_mt, delt_as = [0.04,1.8,0.002]
C, C_err = [0.546,[0.033,-0.033]]

### Misc
wangle, wangle_err = [0.23155,[0.00004,-0.00004]] #sin^2 theta_w
lambda_QCD, QCD_err = [0.2275,[0.01433,-0.01372]]
vev, vev_err = [246,[0,0]]

#Oblique parameters
SOblique, SOblique_err= [0.02,[0.10,-0.10]]
TOblique, TOblique_err=[0.07,[0.12,-0.12]]
UOblique, UOblique_err=[0.00,[0.09,-0.09]]
###################### GLOBAL CONSTRAINT

#m3d = 7.82 #different confidence levels for chisq fit
#nu3 = 14 # degrees of freedom
#
#mHs, tanbs, mAs, chis, chi_min = tdfits(
#        bplus_exp,bplus_err_exp,dplus_exp,dplus_err_exp,dsplus_exp,dsplus_err_exp,
#        bplusmu_exp,bplusmu_err_exp,dsmu_exp,dsmu_err_exp,delt_md,delt_md_err,delt_ms,delt_ms_err,
#        kpi_exp,kpi_exp_err,kpitau_exp,kpitau_exp_err,
#        bs_exp,bs_exp_err,bd_exp,bd_exp_err,rd_exp,rd_exp_err,rdst_exp,rdst_exp_err,
#        branchs,branchs_err,branch_c,branchc_err,gamu,gamu_err,
#        m_u,m_u_err,m_d,m_d_err,m_c,m_c_err,m_s,m_s_err,m_b,m_b_err,mt,mt_err,mW,mW_err,
#        m_bplus,m_bplus_err,m_dplus,m_dplus_err,m_dsplus,m_dsplus_err,mBd,mBd_err,mBs,mBs_err,m_dstar,m_dstar_err,
#        m_K,m_K_err,m_pi,m_pi_err,m_tau,m_tau_err,m_mu,m_mu_err,
#        Vud,Vud_err,Vus,Vus_err,Vub,Vub_err,Vcd,Vcd_err,Vcs,Vcs_err,Vcb,Vcb_err,Vtd,Vtd_err,Vts,Vts_err,Vtb,Vtb_err,
#        etaB,etaB_err,f2Bd,f2Bd_err,f2Bs,f2Bs_err,fBs,fBs_err,BBd,BBd_err,BBs,BBs_err,
#        f_bplus,f_bplus_err,f_dplus,f_dplus_err,f_dsplus,f_dsplus_err,
#        f_Kpi,f_Kpi_err,delt_kpi,delt_kpi_err,delt_kpitau,delt_kpitau_err,
#        tau_bplus,tau_bplus_err,tau_dplus,tau_dplus_err,tau_dsplus,tau_dsplus_err,taubd,taubd_err,taubs,taubs_err,
#        mub,hi,a,A0,ac,at,a_s,B0,bc,bt,b_s,delt_mc,delt_mt,delt_as,1/137,C,C_err,
#        rho,rho_err,rhod,rhod_err,delt_rd,delt_rd_err,r01,r01_err,r11,r11_err,r21,r21_err,
#        delta_b,delta_d,wangle,wangle_err,lambda_QCD,QCD_err,higgs,higgs_err,vev,vev_err,
#        SOblique,SOblique_err,TOblique,TOblique_err,UOblique,UOblique_err,mZ,mZ_err)
#
#hpc, hoc, tbc = chi_del_threed(chi_min[0],chis,mHs,mAs,tanbs,m3d)
#
## print out some numbers to 95CL and 1 sig
#print [10**min(hpc),10**max(hpc)], [10**min(hoc),10**max(hoc)], [10**min(tbc),10**max(tbc)]
#print chi_min[0]/nu3 # reduced chisq, nu = 17 observables - 3 free parameters
#print chi_min
#
#p_val, p_err = p_vals(chi_min[0],nu3)
#print "p-value for global fit with ", nu3," degrees of freedom is: ", "{:.1%}".format(p_val)#, " +/- ", p_err
#
#fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
#ax.scatter(tbc,hoc,hpc,c='darkorchid')
#ax.set_xlabel('$\\log[\\tan\\beta]$')
#ax.set_ylabel('$\\log[m_{A^0}\,(GeV)]$')
#ax.set_zlabel('$\\log[m_{H^+}\,(GeV)]$')
#plt.show()
#
#quit()
#big ol function for everything
mHs, tanbs, chis, chi_2 = itera_global(
        bplus_exp,bplus_err_exp,dplus_exp,dplus_err_exp,dsplus_exp,dsplus_err_exp,
        bplusmu_exp,bplusmu_err_exp,dsmu_exp,dsmu_err_exp,delt_md,delt_md_err,delt_ms,delt_ms_err,
        kpi_exp,kpi_exp_err,kpitau_exp,kpitau_exp_err,
        bs_exp,bs_exp_err,bd_exp,bd_exp_err,rd_exp,rd_exp_err,rdst_exp,rdst_exp_err,
        branchs,branchs_err,branch_c,branchc_err,gamu,gamu_err,
        m_u,m_u_err,m_d,m_d_err,m_c,m_c_err,m_s,m_s_err,m_b,m_b_err,mt,mt_err,mW,mW_err,
        m_bplus,m_bplus_err,m_dplus,m_dplus_err,m_dsplus,m_dsplus_err,mBd,mBd_err,mBs,mBs_err,m_dstar,m_dstar_err,
        m_K,m_K_err,m_pi,m_pi_err,m_tau,m_tau_err,m_mu,m_mu_err,
        Vud,Vud_err,Vus,Vus_err,Vub,Vub_err,Vcd,Vcd_err,Vcs,Vcs_err,Vcb,Vcb_err,Vtd,Vtd_err,Vts,Vts_err,Vtb,Vtb_err,
        etaB,etaB_err,f2Bd,f2Bd_err,f2Bs,f2Bs_err,fBs,fBs_err,BBd,BBd_err,BBs,BBs_err,
        f_bplus,f_bplus_err,f_dplus,f_dplus_err,f_dsplus,f_dsplus_err,
        f_Kpi,f_Kpi_err,delt_kpi,delt_kpi_err,delt_kpitau,delt_kpitau_err,
        tau_bplus,tau_bplus_err,tau_dplus,tau_dplus_err,tau_dsplus,tau_dsplus_err,taubd,taubd_err,taubs,taubs_err,
        mub,hi,a,A0,ac,at,a_s,B0,bc,bt,b_s,delt_mc,delt_mt,delt_as,1/137,C,C_err,
        rho,rho_err,rhod,rhod_err,delt_rd,delt_rd_err,r01,r01_err,r11,r11_err,r21,r21_err,
        delta_b,delta_d,wangle,wangle_err,lambda_QCD,QCD_err,higgs,higgs_err,vev,vev_err,
        SOblique,SOblique_err,TOblique,TOblique_err,UOblique,UOblique_err,mZ,mZ_err)

#h - mH; t - tanb
#l - (semi-)leptonics; b - b mixing; g - bsgamma; a - combine l,b,g; mu - Bs to mumu; l2 -everything
hl, hb, hg, ha, hmu, hl2, hS, hT, hU = mHs
tl, tb, tg, ta, tmu, tl2, tS, tT, tU = tanbs
chi_ls, chi_ms, chi_gs, chi_as, chi_mus, chi_2s, chi_Ss, chi_Ts, chi_Us = chis #all chisq values

m1,m2 = 2.30,5.99 #different confidence levels for chisq fit
nu = 15 # degrees of freedom

# returns mH, tanb values, and 'edges' which is to draw around the perimeters of regions
hchi_leps, tchi_leps, lep_edges_e = chi_del(min(chi_ls),chi_ls,hl,tl,m2) #chisq fit to 95% CL
hchi_leps2, tchi_leps2, lep_edges = chi_del(min(chi_ls),chi_ls,hl,tl,m1) #chisq fit to 1 sigma

hchi_mix, tchi_mix, mix_edges_e = chi_del(min(chi_ms),chi_ms,hb,tb,m2)
hchi_mix2, tchi_mix2, mix_edges = chi_del(min(chi_ms),chi_ms,hb,tb,m1)

hchi_gam, tchi_gam, gam_edges_e = chi_del(min(chi_gs),chi_gs,hg,tg,m2)
hchi_gam2, tchi_gam2, gam_edges = chi_del(min(chi_gs),chi_gs,hg,tg,m1)

hchi_mu, tchi_mu, mu_edges_e = chi_del(min(chi_mus),chi_mus,hmu,tmu,m2)
hchi_mu2, tchi_mu2, mu_edges = chi_del(min(chi_mus),chi_mus,hmu,tmu,m1)

#hchi_S, tchi_S, S_edges_e = chi_del(min(chi_Ss),chi_Ss,hS,tS,m2)
#hchi_S2, tchi_S2, S_edges = chi_del(min(chi_Ss),chi_Ss,hS,tS,m1)

#hchi_T, tchi_T, T_edges_e = chi_del(min(chi_Ts),chi_Ts,hT,tT,m2)
#hchi_T2, tchi_T2, T_edges = chi_del(min(chi_Ts),chi_Ts,hT,tT,m1)

hchi_U, tchi_U, U_edges_e = chi_del(min(chi_Us),chi_Us,hU,tU,m2)
hchi_U2, tchi_U2, U_edges = chi_del(min(chi_Us),chi_Us,hU,tU,m1)

hchi_2, tchi_2, two_edges_e = chi_del(chi_2[0],chi_2s,hl2,tl2,m2)
hchi_22, tchi_22, two_edges = chi_del(chi_2[0],chi_2s,hl2,tl2,m1)

# print out some numbers to 95CL and 1 sig
print [10**min(hchi_2),10**max(hchi_2)], [10**min(hchi_22),10**max(hchi_22)]
print [10**min(tchi_2),10**max(tchi_2)], [10**min(tchi_22), 10**max(tchi_22)]
print chi_2[0]/nu # reduced chisq, nu = 16 observables - 2 free parameters
print chi_2

p_val, p_err = p_vals(chi_2[0],nu)
print "p-value for global fit with ", nu," degrees of freedom is: ", "{:.1%}".format(p_val)#, " +/- ", p_err

# plotting! scatter plots for regions, then the for loops plot out the outline
# all regions plotted to 95% CL, and their lines plotted to this too, but then the 1 sigma everything region's border is plotted
plt.figure(figsize=(8,6))
plt.scatter(tchi_mix,hchi_mix,c='cornflowerblue')
for i, j in mix_edges_e[0]:
    plt.plot(mix_edges_e[1][[i,j],0],mix_edges_e[1][[i,j],1],c='midnightblue',linestyle='--')
plt.scatter(tchi_leps,hchi_leps,c='green')
for i, j in lep_edges_e[0]:
    plt.plot(lep_edges_e[1][[i,j],0],lep_edges_e[1][[i,j],1],c='darkgreen',linestyle='--')
plt.scatter(tchi_gam,hchi_gam,c='coral')
for i, j in gam_edges_e[0]:
    plt.plot(gam_edges_e[1][[i,j],0],gam_edges_e[1][[i,j],1],c='darkgoldenrod',linestyle='--')
plt.scatter(tchi_mu,hchi_mu,c='red')
for i, j in mu_edges_e[0]:
    plt.plot(mu_edges_e[1][[i,j],0],mu_edges_e[1][[i,j],1],c='chocolate',linestyle='--')
plt.scatter(tchi_U,hchi_U,c='orchid')
for i, j in U_edges_e[0]:
    plt.plot(U_edges_e[1][[i,j],0],U_edges_e[1][[i,j],1],c='deeppink',linestyle='--')
plt.scatter(tchi_2,hchi_2,c='darkorchid')
for i, j in two_edges[0]:
    plt.plot(two_edges[1][[i,j],0],two_edges[1][[i,j],1],c='plum',linestyle='--')
plt.axis([-1,2,1,3.5])
plt.ylabel('$\\log[m_{H+}$, GeV]',fontsize=18)
plt.xlabel('$\\log[\\tan(\\beta)]$',fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
#plt.axvline(x=np.log10(6),color='lightseagreen')
plt.axvline(x=np.log10(mt/m_b),color='black')
plt.axhline(y=np.log10(160),color='black',linestyle='--')
plt.annotate('$M\\to l\\nu+\\tau\\to M\\nu$ \n $+ \\mathcal{R}(D)$',xy=(0.05,0.3),xycoords='axes fraction',fontsize=18)
#plt.annotate('$M\\to l\\nu+\\tau\\to M\\nu$ \n $+ \\mathcal{R}(D^{(*)})$',xy=(0.05,0.3),xycoords='axes fraction',fontsize=18)
plt.annotate('$\\Delta M_q$',xy=(0.75,0.15),xycoords='axes fraction',fontsize=18)
plt.annotate('$b\\to s\\gamma$',xy=(0.05,0.9),xycoords='axes fraction',fontsize=18)
plt.annotate('All',xy=(0.65,0.8),xycoords='axes fraction',fontsize=18)
plt.annotate('S,T,U',xy=(0.05,0.73),xycoords='axes fraction',fontsize=13)
plt.annotate('$B_q \\to \\mu^+\\mu^-$',xy=(0.5,0.55),xycoords='axes fraction',fontsize=18)
plt.title('$M = 750\,GeV,\; m_{A^0} = 1.5\,TeV,$ \n $m_{H^0} = 1.5\,TeV,\; \\cos(\\beta-\\alpha) = 0$',fontsize=18)
#plt.title('$M = 750\,GeV,\; m_{H^0} = 1.5\,TeV,\; \\cos(\\beta-\\alpha) = \\sin(2\\beta)$',fontsize=18)
plt.savefig('global_test3sig.png')
#plt.show()
os.system('play draco.mp3')
