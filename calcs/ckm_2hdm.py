from __future__ import division
import numpy as np
from functions import rh
from rdstarring import *

g_gev = (1.1663787e-5)**2
hbar_gev = 6.582119514e-25
g_mev = (1.1663787e-11)**2
hbar_mev = 6.582119514e-22

def v2(vs):
    v = 1
    for i in range(len(vs)):
        v -= vs[i]**2
    return v

def Vp2(vs,vse):
    sigma=1.96
    v = v2(vs)
    u,l=0,0
    for i in range(len(vs)):
        u += pow((2*vse[i][0])/vs[i],2)
        l += pow((2*vse[i][1])/vs[i],2)
    u = sigma*v*np.sqrt(u)
    l = sigma*v*np.sqrt(l)
    return v, u, l

def vs(V,mu,md,mm,tanb,mH):
    '''
        an explanation
    '''
    vs = V/pow(1+rh(mu,md,mm,tanb,mH),2)
    return vs

def error_vs(V,V_err,mu,mu_err,md,md_err,mm,mm_err,tanb,mH):
    '''
        an explanation
    '''
    vst = vs(V,mu,md,mm,tanb,mH)
    err1_up = abs(vs(V+V_err[0],mu,md,mm,tanb,mH)-vst)
    err1_low = abs(vs(V+V_err[1],mu,md,mm,tanb,mH)-vst)
    err2_up = abs(vs(V,mu+mu_err[0],md,mm,tanb,mH)-vst)
    err2_low = abs(vs(V,mu+mu_err[1],md,mm,tanb,mH)-vst)
    err3_up = abs(vs(V,mu,md+md_err[0],mm,tanb,mH)-vst)
    err3_low = abs(vs(V,mu,md+md_err[1],mm,tanb,mH)-vst)
    err4_up = abs(vs(V,mu,md,mm+mm_err[0],tanb,mH)-vst)
    err4_low = abs(vs(V,mu,md,mm+mm_err[1],tanb,mH)-vst)

    upper = np.sqrt(err1_up**2 + err2_up**2 + err3_up**2 + err4_up**2)
    lower = np.sqrt(err1_low**2 + err2_low**2 + err3_low**2 + err4_low**2)

    return upper, lower


def ckmelsr(
        V,V_err,
        mu,mu_err,md,md_err,ms,ms_err,mc,mc_err,mb,mb_err,
        mbp,mbp_err,mdp,mdp_err,mds,mds_err,mtau,mtau_err,mmu,mmu_err,
        mK,mK_err,mpi,mpi_err,
        mBs,mBs_err,mDst,mDst_err,
        rhod,rhod_err,delta,delta_err,vev,vev_err):
    '''
        an explanation
    '''
    Vud, Vus, Vub, Vcd, Vcs, Vcb = V
    Vud_err, Vus_err, Vub_err, Vcd_err, Vcs_err, Vcb_err = V_err
    log_mH_range = np.linspace(0,3.5,350)
    log_tanb_range = np.linspace(-1,2,300)
    mH_range = 10**log_mH_range
    tanb_range = 10**log_tanb_range 
    mH1_loc,tanb1_loc,val1_loc = [],[],[]
    mH2_loc,tanb2_loc,val2_loc = [],[],[]
    mHa_loc,tanba_loc,vala_loc = [],[],[]
    for i in mH_range:
        for j in tanb_range:
            expect_vub = vs(Vub,mu,mb,mbp,j,i)
            expect_vub_err = error_vs(Vub,Vub_err,mu,mu_err,mb,mb_err,mbp,mbp_err,j,i)

            expect_vcd = vs(Vcd,mc,md,mdp,j,i)
            expect_vcd_err = error_vs(Vcd,Vcd_err,mc,mc_err,md,md_err,mdp,mdp_err,j,i)

            expect_vcs = vs(Vcs,mc,ms,mds,j,i)
            expect_vcs_err = error_vs(Vcs,Vcs_err,mc,mc_err,ms,ms_err,mds,mds_err,j,i)

            expect_vus = vs(Vus,mu,ms,mK,j,i)
            expect_vus_err = error_vs(Vus,Vus_err,mu,mu_err,ms,ms_err,mK,mK_err,j,i)

#            Vcb_e, Vcb_erre = error_rdn(mBs,mBs_err,mdp,mdp_err,rhod,rhod_err,delta,delta_err,Vcb,Vcb_err,mmu,mmu_err,mtau,mtau_err,vev,vev_err,mc,mc_err,mb,mb_err,j,i)

            row1, u1, l1 = Vp2([Vud,expect_vus,expect_vub],[Vud_err,expect_vus_err,expect_vub_err])
            row2, u2, l2 = Vp2([expect_vcd,expect_vcs,Vcb],[expect_vcd_err,expect_vcs_err,Vcb_err])
            r1_bool = (row1 > 0) or (row1+u1 > 0) or (row1-l1 > 0)
            r2_bool = (row2 > 0) or (row2+u2 > 0) or (row2-l2 > 0)
            if r1_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mH1_loc = np.append(mH1_loc,i_log)
                tanb1_loc = np.append(tanb1_loc,j_log)
                val1_loc = np.append(val1_loc,row1)
            if r2_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mH2_loc = np.append(mH2_loc,i_log)
                tanb2_loc = np.append(tanb2_loc,j_log)
                val2_loc = np.append(val2_loc,row2)
            if r1_bool and r2_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mHa_loc = np.append(mHa_loc,i_log)
                tanba_loc = np.append(tanba_loc,j_log)
                vala_loc = np.append(vala_loc,[row1,row2])

    return mH1_loc, tanb1_loc, val1_loc, mH2_loc, tanb2_loc, val2_loc, mHa_loc, tanba_loc, vala_loc


    

