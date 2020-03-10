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
#        vs[i] += vse[i][0]
#        u += abs(v2(vs)-v)**2
#        vs[i] += (vse[i][1]-vse[i][0])
#        l += abs(v2(vs)-v)**2
#        vs[i] -= vse[i][1]
    u = sigma*v*np.sqrt(u)
    l = sigma*v*np.sqrt(l)
    return v, u, l

def vsm(mm,ml,fm,taum):
    '''
        Calculates SM branching ratio
    '''
    Bs = (1/(8*np.pi))*(g_gev*mm*ml**2)*((1-(ml**2/mm**2))**2)*(fm**2)*taum*0.982
    return Bs

def vthe(mm,ml,fm,taum,mu,md,tanb,mH,exp):
    '''
        bsm*(1+rh)^2 to check against exp
    '''
    branching = vsm(mm,ml,fm,taum)*(1+rh(mu,md,mm,tanb,mH))**2
    V = np.sqrt(exp/branching)
    return V

def error_vranching(mm,mm_err,ml,ml_err,fm,fm_err,taum,taum_err,mu,mu_err,md,md_err,tanb,mH,exp,exp_err):
    '''
        Calculates errors in branching ratios, using functional method
        - all err vars are [up,low]
    '''
    brt = vthe(mm,ml,fm,taum,mu,md,tanb,mH,exp)
    err1_up = abs(vthe(mm+mm_err[0],ml,fm,taum,mu,md,tanb,mH,exp)-brt)
    err1_low = abs(vthe(mm+mm_err[1],ml,fm,taum,mu,md,tanb,mH,exp)-brt)
    err2_up = abs(vthe(mm,ml+ml_err[0],fm,taum,mu,md,tanb,mH,exp)-brt)
    err2_low = abs(vthe(mm,ml+ml_err[1],fm,taum,mu,md,tanb,mH,exp)-brt)
    err3_up = abs(vthe(mm,ml,fm,taum,mu,md,tanb,mH,exp+exp_err[0])-brt)
    err3_low = abs(vthe(mm,ml,fm,taum,mu,md,tanb,mH,exp+exp_err[1])-brt)
    err4_up = abs(vthe(mm,ml,fm+fm_err[0],taum,mu,md,tanb,mH,exp)-brt)
    err4_low = abs(vthe(mm,ml,fm+fm_err[1],taum,mu,md,tanb,mH,exp)-brt)
    err5_up = abs(vthe(mm,ml,fm,taum+taum_err[0],mu,md,tanb,mH,exp)-brt)
    err5_low = abs(vthe(mm,ml,fm,taum+taum_err[1],mu,md,tanb,mH,exp)-brt)
    err6_up = abs(vthe(mm,ml,fm,taum,mu+mu_err[0],md,tanb,mH,exp)-brt)
    err6_low = abs(vthe(mm,ml,fm,taum,mu+mu_err[1],md,tanb,mH,exp)-brt)
    err7_up = abs(vthe(mm,ml,fm,taum,mu,md+md_err[0],tanb,mH,exp)-brt)
    err7_low = abs(vthe(mm,ml,fm,taum,mu,md+md_err[1],tanb,mH,exp)-brt)

    upper = np.sqrt(err1_up**2 + err2_up**2 + err3_up**2 + err4_up**2 + err5_up**2 + err6_up**2 + err7_up**2)
    lower = np.sqrt(err1_low**2 + err2_low**2 + err3_low**2 + err4_low**2 + err5_low**2 + err6_low**2 + err7_low**2)

    return upper, lower

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


def vratios(mK,mpi,ml,mtau,fKpi,delt_kpi,delt_tau):
    '''
        Decay ratio function for Kaon and pion leptonic partial widths - easier than full branching fractions
    '''

    kpi = (mK/mpi)*(((1-(ml**2)/(mK**2))/(1-(ml**2)/(mpi**2)))**2)*(fKpi**2)*(1+delt_kpi)
    tau_kpi = (((1-(mK**2)/(mtau**2))/(1-(mpi**2)/(mtau**2)))**2)*(fKpi**2)*(1+delt_tau)

    return kpi, tau_kpi

def decay_bsm(mK,mpi,ml,mtau,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH,ke,te,Vud):
    '''
        Extend ratios to 2HDM
    '''
    kpi_sm, tau_kpi_sm = vratios(mK,mpi,ml,mtau,fKpi,delt_kpi,delt_tau)
    rh1 = rh(mu,ms,mK,tanb,mH)
    rh2 = rh(mu,md,mpi,tanb,mH)
    rat = ((1+rh1)**2)/((1+rh2)**2)
    kpi_bsm = kpi_sm*rat
    tau_kpi_bsm = tau_kpi_sm*rat
    Vus1 = np.sqrt(ke*Vud/kpi_bsm)
    Vus2 = np.sqrt(te*Vud/tau_kpi_bsm)
    Vus = (Vus1+Vus2)/2

    return Vus

def error_kpi(mK,mK_err,mpi,mpi_err,ml,ml_err,mtau,mtau_err,fKpi,fKpi_err,delt_kpi,delt_kpi_err,delt_tau,delt_tau_err,ms,ms_err,md,md_err,mu,mu_err,tanb,mH,ke,ke_err,te,te_err,Vud,Vud_err):
    '''
        Error propagation for kpi ratios
    '''
    kpi = decay_bsm(mK,mpi,ml,mtau,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH,ke,te,Vud)

    ## errors
    err1_up = decay_bsm(mK+mK_err[0],mpi,ml,mtau,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH,ke,te,Vud)
    err1_low = decay_bsm(mK+mK_err[1],mpi,ml,mtau,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH,ke,te,Vud)
    err2_up = decay_bsm(mK,mpi+mpi_err[0],ml,mtau,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH,ke,te,Vud)
    err2_low = decay_bsm(mK,mpi+mpi_err[1],ml,mtau,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH,ke,te,Vud)
    err3_up = decay_bsm(mK,mpi,ml+ml_err[0],mtau,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH,ke,te,Vud)
    err3_low = decay_bsm(mK,mpi,ml+ml_err[1],mtau,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH,ke,te,Vud)
    err4_up = decay_bsm(mK,mpi,ml,mtau+mtau_err[0],fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH,ke,te,Vud)
    err4_low = decay_bsm(mK,mpi,ml,mtau+mtau_err[1],fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH,ke,te,Vud)
    err5_up = decay_bsm(mK,mpi,ml,mtau,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH,ke+ke_err[0],te,Vud)
    err5_low = decay_bsm(mK,mpi,ml,mtau,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH,ke+ke_err[1],te,Vud)
    err6_up = decay_bsm(mK,mpi,ml,mtau,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH,ke,te+te_err[0],Vud)
    err6_low = decay_bsm(mK,mpi,ml,mtau,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH,ke,te+te_err[1],Vud)
    err7_up = decay_bsm(mK,mpi,ml,mtau,fKpi+fKpi_err[0],delt_kpi,delt_tau,ms,md,mu,tanb,mH,ke,te,Vud)
    err7_low = decay_bsm(mK,mpi,ml,mtau,fKpi+fKpi_err[1],delt_kpi,delt_tau,ms,md,mu,tanb,mH,ke,te,Vud)
    err9_up = decay_bsm(mK,mpi,ml,mtau,fKpi,delt_kpi+delt_kpi_err[0],delt_tau,ms,md,mu,tanb,mH,ke,te,Vud)
    err9_low = decay_bsm(mK,mpi,ml,mtau,fKpi,delt_kpi+delt_kpi_err[1],delt_tau,ms,md,mu,tanb,mH,ke,te,Vud)
    err10_up = decay_bsm(mK,mpi,ml,mtau,fKpi,delt_kpi,delt_tau+delt_tau_err[0],ms,md,mu,tanb,mH,ke,te,Vud)
    err10_low = decay_bsm(mK,mpi,ml,mtau,fKpi,delt_kpi,delt_tau+delt_tau_err[1],ms,md,mu,tanb,mH,ke,te,Vud)
    err11_up = decay_bsm(mK,mpi,ml,mtau,fKpi,delt_kpi,delt_tau,ms+ms_err[0],md,mu,tanb,mH,ke,te,Vud)
    err11_low = decay_bsm(mK,mpi,ml,mtau,fKpi,delt_kpi,delt_tau,ms+ms_err[1],md,mu,tanb,mH,ke,te,Vud)
    err12_up = decay_bsm(mK,mpi,ml,mtau,fKpi,delt_kpi,delt_tau,ms,md+md_err[0],mu,tanb,mH,ke,te,Vud)
    err12_low = decay_bsm(mK,mpi,ml,mtau,fKpi,delt_kpi,delt_tau,ms,md+md_err[1],mu,tanb,mH,ke,te,Vud)
    err13_up = decay_bsm(mK,mpi,ml,mtau,fKpi,delt_kpi,delt_tau,ms,md,mu+mu_err[0],tanb,mH,ke,te,Vud)
    err13_low = decay_bsm(mK,mpi,ml,mtau,fKpi,delt_kpi,delt_tau,ms,md,mu+mu_err[1],tanb,mH,ke,te,Vud)
    err14_up = decay_bsm(mK,mpi,ml,mtau,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH,ke,te,Vud+Vud_err[0])
    err14_low = decay_bsm(mK,mpi,ml,mtau,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH,ke,te,Vud+Vud_err[1])

    ## kpi
    err1_up1, err1_low1 = abs(err1_up-kpi),abs(err1_low-kpi)
    err2_up1, err2_low1 = abs(err2_up-kpi),abs(err2_low-kpi)
    err3_up1, err3_low1 = abs(err3_up-kpi),abs(err3_low-kpi)
    err4_up1, err4_low1 = abs(err4_up-kpi),abs(err4_low-kpi)
    err5_up1, err5_low1 = abs(err5_up-kpi),abs(err5_low-kpi)
    err6_up1, err6_low1 = abs(err6_up-kpi),abs(err6_low-kpi)
    err7_up1, err7_low1 = abs(err7_up-kpi),abs(err7_low-kpi)
    err9_up1, err9_low1 = abs(err9_up-kpi),abs(err9_low-kpi)
    err10_up1, err10_low1 = abs(err10_up-kpi),abs(err10_low-kpi)
    err11_up1, err11_low1 = abs(err11_up-kpi),abs(err11_low-kpi)
    err12_up1, err12_low1 = abs(err12_up-kpi),abs(err12_low-kpi)
    err13_up1, err13_low1 = abs(err13_up-kpi),abs(err13_low-kpi)
    err14_up1, err14_low1 = abs(err14_up-kpi),abs(err14_low-kpi)

    upper1 = np.sqrt(err1_up1**2 + err2_up1**2 + err3_up1**2 + err4_up1**2 + err5_up1**2 + err6_up1**2 + err7_up1**2 + err9_up1**2 + err10_up1**2 + err11_up1**2 + err11_up1**2 + err12_up1**2 + err13_up1**2 + err14_up1**2)
    lower1 = np.sqrt(err1_low1**2 + err2_low1**2 + err3_low1**2 + err4_low1**2 + err5_low1**2 + err6_low1**2 + err7_low1**2 + err9_low1**2 + err10_low1**2 + err11_low1**2 + err11_low1**2 + err12_low1**2 + err13_low1**2 + err14_low1)

    return upper1, lower1

def ckmelsr(V,V_err,mu,mu_err,md,md_err,ms,ms_err,mc,mc_err,mb,mb_err,mbp,mbp_err,mdp,mdp_err,mds,mds_err,mtau,mtau_err,mmu,mmu_err,fb,fb_err,fd,fd_err,fds,fds_err,taub,taub_err,taud,taud_err,tauds,tauds_err,expb,expb_err,expd,expd_err,expds,expds_err,mK,mK_err,mpi,mpi_err,fKpi,fKpi_err,delt_kpi,delt_kpi_err,delt_tau,delt_tau_err,ke,ke_err,te,te_err,mBs,mBs_err,mDst,mDst_err,rhod,rhod_err,delta,delta_err,vev,vev_err):
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
            expect_vub = vthe(mbp,mtau,fb,taub,mu,mb,j,i,expb)
            expect_vub_err = error_vranching(mbp,mbp_err,mtau,mtau_err,fb,fb_err,taub,taub_err,mu,mu_err,mb,mb_err,j,i,expb,expb_err)
#            expect_vub_up,expect_vub_down = expect_vub+expect_vub_err[0],expect_vub-expect_vub_err[1]

            expect_vcd = vthe(mdp,mmu,fd,taud,mc,md,j,i,expd)
            expect_vcd_err = error_vranching(mdp,mdp_err,mmu,mmu_err,fd,fd_err,taud,taud_err,mc,mc_err,md,md_err,j,i,expd,expd_err)

            expect_vcs = vthe(mds,mtau,fds,tauds,mc,ms,j,i,expds)
            expect_vcs_err = error_vranching(mds,mds_err,mtau,mtau_err,fds,fds_err,tauds,tauds_err,mc,mc_err,ms,ms_err,j,i,expds,expds_err)

            expect_vus = decay_bsm(mK,mpi,mmu,mtau,fKpi,delt_kpi,delt_tau,ms,md,mu,j,i,ke,te,Vud)
            expect_vus_err = error_kpi(mK,mK_err,mpi,mpi_err,mmu,mmu_err,mtau,mtau_err,fKpi,fKpi_err,delt_kpi,delt_kpi_err,delt_tau,delt_tau_err,ms,ms_err,md,md_err,mu,mu_err,j,i,ke,ke_err,te,te_err,Vud,Vud_err)

            Vcb_erre, Vcb_err = error_rdn(mBs,mBs_err,mDst,mDst_err,rhod,rhod_err,delta,delta_err,Vcb,Vcb_err,mmu,mmu_err,mtau,mtau_err,vev,vev_err,mc,mc_err,mb,mb_err,j,i)

            row1, u1, l1 = Vp2([Vud,expect_vus,expect_vub],[Vud_err,expect_vus_err,expect_vub_err])
            row2, u2, l2 = Vp2([expect_vcd,expect_vcs,Vcb_e],[expect_vcd_err,expect_vcs_err,Vcb_erre])
            r1_bool = (row1 > 0) or (row1+u1 > 0) or (row1-l1 > 0)
            r2_bool = (row2 > 0) or (row2+u2 > 0) or (row2-l2 > 0)
            #ve = vs(V,mu,md,mm,j,i)
            #verr = error_vs(V,V_err,mu,mu_err,md,md_err,mm,mm_err,j,i)
            #ve_up,ve_d = ve+verr[0],ve-verr[1]
            #if (ve >= expect_v and expect_up >= ve_d) or (ve <= expect_v and expect_down <= ve_up):
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


    

