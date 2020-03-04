from __future__ import division 
import numpy as np
from fitting import *
from scipy.integrate import quad

g_gev = (1.1663787e-5)**2
hbar_gev = 6.582119514e-25
g_mev = (1.1663787e-11)**2
hbar_mev = 6.582118514e-22

def bsm(mm,ml,Vud,fm,taum,delta):
    '''
        Calculates SM branching ratio
    '''
    Bs = (1/(8*np.pi))*(g_gev*mm*ml**2)*((1-(ml**2/mm**2))**2)*(Vud**2)*(fm**2)*taum*delta#(0.982*0.99)
    return Bs

def rh(mu,md,mm,tanb,mH):
    '''
        Returns 2HDM correction factor rh
    '''
    r = ((mu-md*tanb**2)/(mu+md))*(mm/mH)**2
    return r

def bthe(mm,ml,Vud,fm,taum,mu,md,tanb,mH,delta):
    '''
        bsm*(1+rh)^2 to check against exp
    '''
    branching = bsm(mm,ml,Vud,fm,taum,delta)*(1+rh(mu,md,mm,tanb,mH))**2
    return branching

def error_branching(mm,mm_err,ml,ml_err,Vud,Vud_err,fm,fm_err,taum,taum_err,mu,mu_err,md,md_err,tanb,mH,delta):
    '''
        Calculates errors in branching ratios, using functional method
        - all err vars are [up,low]
    '''
    brt = bthe(mm,ml,Vud,fm,taum,mu,md,tanb,mH,delta)
    err1_up = abs(bthe(mm+mm_err[0],ml,Vud,fm,taum,mu,md,tanb,mH,delta)-brt)
    err1_low = abs(bthe(mm+mm_err[1],ml,Vud,fm,taum,mu,md,tanb,mH,delta)-brt)
    err2_up = abs(bthe(mm,ml+ml_err[0],Vud,fm,taum,mu,md,tanb,mH,delta)-brt)
    err2_low = abs(bthe(mm,ml+ml_err[1],Vud,fm,taum,mu,md,tanb,mH,delta)-brt)
    err3_up = abs(bthe(mm,ml,Vud+Vud_err[0],fm,taum,mu,md,tanb,mH,delta)-brt)
    err3_low = abs(bthe(mm,ml,Vud+Vud_err[1],fm,taum,mu,md,tanb,mH,delta)-brt)
    err4_up = abs(bthe(mm,ml,Vud,fm+fm_err[0],taum,mu,md,tanb,mH,delta)-brt)
    err4_low = abs(bthe(mm,ml,Vud,fm+fm_err[1],taum,mu,md,tanb,mH,delta)-brt)
    err5_up = abs(bthe(mm,ml,Vud,fm,taum+taum_err[0],mu,md,tanb,mH,delta)-brt)
    err5_low = abs(bthe(mm,ml,Vud,fm,taum+taum_err[1],mu,md,tanb,mH,delta)-brt)
    err6_up = abs(bthe(mm,ml,Vud,fm,taum,mu+mu_err[0],md,tanb,mH,delta)-brt)
    err6_low = abs(bthe(mm,ml,Vud,fm,taum,mu+mu_err[1],md,tanb,mH,delta)-brt)
    err7_up = abs(bthe(mm,ml,Vud,fm,taum,mu,md+md_err[0],tanb,mH,delta)-brt)
    err7_low = abs(bthe(mm,ml,Vud,fm,taum,mu,md+md_err[1],tanb,mH,delta)-brt)

    upper = np.sqrt(err1_up**2 + err2_up**2 + err3_up**2 + err4_up**2 + err5_up**2 + err6_up**2 + err7_up**2)
    lower = np.sqrt(err1_low**2 + err2_low**2 + err3_low**2 + err4_low**2 + err5_low**2 + err6_low**2 + err7_low**2)

    return upper, lower

def itera(mm,mm_err,ml,ml_err,Vud,Vud_err,fm,fm_err,taum,taum_err,mu,mu_err,md,md_err,branch_exp,branch_exp_error,delta):
    '''
        Choose min,max limits for scope of tan(beta) and mH+, then check for each point in this if:
            - upper error on branching sm is above the lower error on branching exp
            - lower error on branching sm is below the upper error on branching exp
        If either is true, plot a point at coordinate, and tadaa
    '''
    sigma = 1
    exp_branch_up,exp_branch_down = branch_exp+branch_exp_error[0],branch_exp+branch_exp_error[1]
    av_br = 0.5*(exp_branch_up+exp_branch_down)
    sige_br = sigma*(exp_branch_up-av_br)
    log_mH_range = np.linspace(0,3.5,350)
    log_tanb_range = np.linspace(-1,2,300)
    mH_range = 10**log_mH_range
    tanb_range = 10**log_tanb_range
    mH_loc = []
    tanb_loc = []
    for i in mH_range:
        for j in tanb_range:
            expect_branch = bthe(mm,ml,Vud,fm,taum,mu,md,j,i,delta)
            expect_error = error_branching(mm,mm_err,ml,ml_err,Vud,Vud_err,fm,fm_err,taum,taum_err,mu,mu_err,md,md_err,j,i,delta)
            expect_branch_up, expect_branch_down = expect_branch+expect_error[0],expect_branch-expect_error[1]
            print expect_branch_down
            mid_br = 0.5*(expect_branch_up+expect_branch_down)
            sig_br = sigma*(expect_branch_up-mid_br)
            br_bool = ((av_br >= mid_br and mid_br+sig_br >= av_br-sige_br) or (av_br <= mid_br and mid_br-sig_br <= av_br+sige_br)) 
            if br_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mH_loc = np.append(mH_loc,i_log)
                tanb_loc = np.append(tanb_loc,j_log)

    return mH_loc, tanb_loc

def mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq,expect):
    '''
        B mixing mass eqn
    '''
    x_tH = mt**2/mH**2
    x_tW = mt**2/mW**2
    x_HW = mH**2/mW**2
    S_WH = (x_tH/tanb**2)*((2*x_HW-8)*np.log(x_tH)/((1-x_HW)*(1-x_tH)**2) + 6*x_HW*np.log(x_tW)/((1-x_HW)*(1-x_tW)**2) - (8-2*x_tW)/((1-x_tW)*(1-x_tH)))
    S_WW = (1 + 9/(1-x_tW) - 6/((1-x_tW)**2) - 6*(x_tW**2)*np.log(x_tW)/((1-x_tW)**3))
    S_HH = (x_tH/tanb**4)*((1+x_tH)/((1-x_tH)**2) + 2*x_tH*np.log(x_tH)/((1-x_tH)**3))

    #delt_mq = g_mev/(24*np.pi**2)*((Vtq*Vtb)**2)*etaB*mB*(mt**2)*(fBq**2)*BBq*(S_WW + S_WH + S_HH)
    delt_mq = expect*(1+(S_WH/S_WW)+(S_HH/S_WW))

    return delt_mq#/hbar_mev

def error_mixing(mt,mt_err,mH,mW,mW_err,tanb,Vtq,Vtq_err,Vtb,Vtb_err,etaB,etaB_err,mB,mB_err,fBq,fBq_err,BBq,BBq_err,expect,expect_err):
    '''
        Calculates errors in branching ratios, using functional method
        - all err vars are [up,low]
    '''
    mix = mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq,expect)
    err1_up = abs(mixing(mt+mt_err[0],mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq,expect)-mix)
    err1_low = abs(mixing(mt+mt_err[1],mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq,expect)-mix)
    err2_up = abs(mixing(mt,mH,mW+mW_err[0],tanb,Vtq,Vtb,etaB,mB,fBq,BBq,expect)-mix)
    err2_low = abs(mixing(mt,mH,mW+mW_err[1],tanb,Vtq,Vtb,etaB,mB,fBq,BBq,expect)-mix)
    err3_up = abs(mixing(mt,mH,mW,tanb,Vtq+Vtq_err[0],Vtb,etaB,mB,fBq,BBq,expect)-mix)
    err3_low = abs(mixing(mt,mH,mW,tanb,Vtq+Vtq_err[1],Vtb,etaB,mB,fBq,BBq,expect)-mix)
    err4_up = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb+Vtb_err[0],etaB,mB,fBq,BBq,expect)-mix)
    err4_low = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb+Vtb_err[1],etaB,mB,fBq,BBq,expect)-mix)
    err5_up = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB+etaB_err[0],mB,fBq,BBq,expect)-mix)
    err5_low = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB+etaB_err[1],mB,fBq,BBq,expect)-mix)
    err6_up = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB+mB_err[0],fBq,BBq,expect)-mix)
    err6_low = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB+mB_err[1],fBq,BBq,expect)-mix)
    err7_up = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq+fBq_err[0],BBq,expect)-mix)
    err7_low = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq+fBq_err[1],BBq,expect)-mix)
    err8_up = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq+BBq_err[0],expect)-mix)
    err8_low = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq+BBq_err[1],expect)-mix)
    err9_up = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq+BBq_err[0],expect+expect_err[0])-mix)
    err9_low = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq+BBq_err[1],expect+expect_err[1])-mix)

    upper = np.sqrt(err1_up**2 + err2_up**2 + err3_up**2 + err4_up**2 + err5_up**2 + err6_up**2 + err7_up**2 + err8_up**2 + err9_up**2)
    lower = np.sqrt(err1_low**2 + err2_low**2 + err3_low**2 + err4_low**2 + err5_low**2 + err6_low**2 + err7_low**2 + err8_low**2 + err9_low**2)

    return upper, lower

def itera_mix(mt,mt_err,mW,mW_err,Vtq,Vtq_err,Vtb,Vtb_err,etaB,etaB_err,mB,mB_err,fBq,fBq_err,BBq,BBq_err,branch_exp,branch_exp_error,expect,expect_err):
    '''
        Choose min,max limits for scope of tan(beta) and mH+, then check for each point in this if:
            - upper error on branching sm is above the lower error on branching exp
            - lower error on branching sm is below the upper error on branching exp
        If either is true, plot a point at coordinate, and tadaa
    '''
    sigma = 1.96
    exp_branch_up,exp_branch_down = branch_exp+branch_exp_error[0],branch_exp+branch_exp_error[1]
    av_br = 0.5*(exp_branch_up+exp_branch_down)
    sige_br = sigma*(exp_branch_up-av_br)
    log_mH_range = np.linspace(3,6.5,350)
    log_tanb_range = np.linspace(-1,2,300)
    mH_range = 10**log_mH_range
    tanb_range = 10**log_tanb_range
    mH_loc = []
    tanb_loc = []
    for i in mH_range:
        for j in tanb_range:
            expect_branch = mixing(mt,i,mW,j,Vtq,Vtb,etaB,mB,fBq,BBq,expect)
#            print expect_branch*1e-12
            expect_error = error_mixing(mt,mt_err,i,mW,mW_err,j,Vtq,Vtq_err,Vtb,Vtb_err,etaB,etaB_err,mB,mB_err,fBq,fBq_err,BBq,BBq_err,expect,expect_err)
            expect_branch_up, expect_branch_down = expect_branch+expect_error[0],expect_branch-expect_error[1]
            mid_br = 0.5*(expect_branch_up+expect_branch_down)
            sig_br = sigma*(expect_branch_up-mid_br)
            br_bool = ((av_br >= mid_br and mid_br+sig_br >= av_br-sige_br) or (av_br <= mid_br and mid_br-sig_br <= av_br+sige_br)) 
            if br_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mH_loc = np.append(mH_loc,i_log)
                tanb_loc = np.append(tanb_loc,j_log)

    for i in range(len(mH_loc)):
        mH_loc[i] = mH_loc[i]-3

    return mH_loc, tanb_loc

def decay_ratios(mK,mpi,ml,mtau,Vus,Vud,fKpi,delt_kpi,delt_tau):
    '''
        Decay ratio function for Kaon and pion leptonic partial widths - easier than full branching fractions
    '''

    kpi = (mK/mpi)*(((1-(ml**2)/(mK**2))/(1-(ml**2)/(mpi**2)))**2)*((Vus/Vud)**2)*(fKpi**2)*(1+delt_kpi)
    tau_kpi = (((1-(mK**2)/(mtau**2))/(1-(mpi**2)/(mtau**2)))**2)*((Vus/Vud)**2)*(fKpi**2)*(1+delt_tau)

    return kpi, tau_kpi

def decay_bsm(mK,mpi,ml,mtau,Vus,Vud,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH):
    '''
        Extend ratios to 2HDM
    '''
    kpi_sm, tau_kpi_sm = decay_ratios(mK,mpi,ml,mtau,Vus,Vud,fKpi,delt_kpi,delt_tau)
    rh1 = rh(mu,ms,mK,tanb,mH)
    rh2 = rh(mu,md,mpi,tanb,mH)
    rat = ((1+rh1)**2)/((1+rh2)**2)
    kpi_bsm = kpi_sm*rat
    tau_kpi_bsm = tau_kpi_sm*rat

    return kpi_bsm, tau_kpi_bsm

def error_kpi(mK,mK_err,mpi,mpi_err,ml,ml_err,mtau,mtau_err,Vus,Vus_err,Vud,Vud_err,fKpi,fKpi_err,delt_kpi,delt_kpi_err,delt_tau,delt_tau_err,ms,ms_err,md,md_err,mu,mu_err,tanb,mH):
    '''
        Error propagation for kpi ratios
    '''
    kpi, kpi_tau = decay_bsm(mK,mpi,ml,mtau,Vus,Vud,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH)

    ## errors
    err1_up = decay_bsm(mK+mK_err[0],mpi,ml,mtau,Vus,Vud,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH)
    err1_low = decay_bsm(mK+mK_err[1],mpi,ml,mtau,Vus,Vud,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH)
    err2_up = decay_bsm(mK,mpi+mpi_err[0],ml,mtau,Vus,Vud,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH)
    err2_low = decay_bsm(mK,mpi+mpi_err[1],ml,mtau,Vus,Vud,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH)
    err3_up = decay_bsm(mK,mpi,ml+ml_err[0],mtau,Vus,Vud,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH)
    err3_low = decay_bsm(mK,mpi,ml+ml_err[1],mtau,Vus,Vud,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH)
    err4_up = decay_bsm(mK,mpi,ml,mtau+mtau_err[0],Vus,Vud,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH)
    err4_low = decay_bsm(mK,mpi,ml,mtau+mtau_err[1],Vus,Vud,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH)
    err5_up = decay_bsm(mK,mpi,ml,mtau,Vus+Vus_err[0],Vud,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH)
    err5_low = decay_bsm(mK,mpi,ml,mtau,Vus+Vus_err[1],Vud,fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH)
    err6_up = decay_bsm(mK,mpi,ml,mtau,Vus,Vud+Vud_err[0],fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH)
    err6_low = decay_bsm(mK,mpi,ml,mtau,Vus,Vud+Vud_err[1],fKpi,delt_kpi,delt_tau,ms,md,mu,tanb,mH)
    err7_up = decay_bsm(mK,mpi,ml,mtau,Vus,Vud,fKpi+fKpi_err[0],delt_kpi,delt_tau,ms,md,mu,tanb,mH)
    err7_low = decay_bsm(mK,mpi,ml,mtau,Vus,Vud,fKpi+fKpi_err[1],delt_kpi,delt_tau,ms,md,mu,tanb,mH)
    err9_up = decay_bsm(mK,mpi,ml,mtau,Vus,Vud,fKpi,delt_kpi+delt_kpi_err[0],delt_tau,ms,md,mu,tanb,mH)
    err9_low = decay_bsm(mK,mpi,ml,mtau,Vus,Vud,fKpi,delt_kpi+delt_kpi_err[1],delt_tau,ms,md,mu,tanb,mH)
    err10_up = decay_bsm(mK,mpi,ml,mtau,Vus,Vud,fKpi,delt_kpi,delt_tau+delt_tau_err[0],ms,md,mu,tanb,mH)
    err10_low = decay_bsm(mK,mpi,ml,mtau,Vus,Vud,fKpi,delt_kpi,delt_tau+delt_tau_err[1],ms,md,mu,tanb,mH)
    err11_up = decay_bsm(mK,mpi,ml,mtau,Vus,Vud,fKpi,delt_kpi,delt_tau,ms+ms_err[0],md,mu,tanb,mH)
    err11_low = decay_bsm(mK,mpi,ml,mtau,Vus,Vud,fKpi,delt_kpi,delt_tau,ms+ms_err[1],md,mu,tanb,mH)
    err12_up = decay_bsm(mK,mpi,ml,mtau,Vus,Vud,fKpi,delt_kpi,delt_tau,ms,md+md_err[0],mu,tanb,mH)
    err12_low = decay_bsm(mK,mpi,ml,mtau,Vus,Vud,fKpi,delt_kpi,delt_tau,ms,md+md_err[1],mu,tanb,mH)
    err13_up = decay_bsm(mK,mpi,ml,mtau,Vus,Vud,fKpi,delt_kpi,delt_tau,ms,md,mu+mu_err[0],tanb,mH)
    err13_low = decay_bsm(mK,mpi,ml,mtau,Vus,Vud,fKpi,delt_kpi,delt_tau,ms,md,mu+mu_err[1],tanb,mH)

    ## kpi
    err1_up1, err1_low1 = abs(err1_up[0]-kpi),abs(err1_low[0]-kpi)
    err2_up1, err2_low1 = abs(err2_up[0]-kpi),abs(err2_low[0]-kpi)
    err3_up1, err3_low1 = abs(err3_up[0]-kpi),abs(err3_low[0]-kpi)
    err4_up1, err4_low1 = abs(err4_up[0]-kpi),abs(err4_low[0]-kpi)
    err5_up1, err5_low1 = abs(err5_up[0]-kpi),abs(err5_low[0]-kpi)
    err6_up1, err6_low1 = abs(err6_up[0]-kpi),abs(err6_low[0]-kpi)
    err7_up1, err7_low1 = abs(err7_up[0]-kpi),abs(err7_low[0]-kpi)
    err9_up1, err9_low1 = abs(err9_up[0]-kpi),abs(err9_low[0]-kpi)
    err10_up1, err10_low1 = abs(err10_up[0]-kpi),abs(err10_low[0]-kpi)
    err11_up1, err11_low1 = abs(err11_up[0]-kpi),abs(err11_low[0]-kpi)
    err12_up1, err12_low1 = abs(err12_up[0]-kpi),abs(err12_low[0]-kpi)
    err13_up1, err13_low1 = abs(err13_up[0]-kpi),abs(err13_low[0]-kpi)

    upper1 = np.sqrt(err1_up1**2 + err2_up1**2 + err3_up1**2 + err4_up1**2 + err5_up1**2 + err6_up1**2 + err7_up1**2 + err9_up1**2 + err10_up1**2 + err11_up1**2 + err11_up1**2 + err12_up1**2 + err13_up1**2)
    lower1 = np.sqrt(err1_low1**2 + err2_low1**2 + err3_low1**2 + err4_low1**2 + err5_low1**2 + err6_low1**2 + err7_low1**2 + err9_low1**2 + err10_low1**2 + err11_low1**2 + err11_low1**2 + err12_low1**2 + err13_low1**2)

    ## tau_kpi
    err1_up2, err1_low2 = abs(err1_up[1]-kpi_tau),abs(err1_low[1]-kpi_tau)
    err2_up2, err2_low2 = abs(err2_up[1]-kpi_tau),abs(err2_low[1]-kpi_tau)
    err3_up2, err3_low2 = abs(err3_up[1]-kpi_tau),abs(err3_low[1]-kpi_tau)
    err4_up2, err4_low2 = abs(err4_up[1]-kpi_tau),abs(err4_low[1]-kpi_tau)
    err5_up2, err5_low2 = abs(err5_up[1]-kpi_tau),abs(err5_low[1]-kpi_tau)
    err6_up2, err6_low2 = abs(err6_up[1]-kpi_tau),abs(err6_low[1]-kpi_tau)
    err7_up2, err7_low2 = abs(err7_up[1]-kpi_tau),abs(err7_low[1]-kpi_tau)
    err9_up2, err9_low2 = abs(err9_up[1]-kpi_tau),abs(err9_low[1]-kpi_tau)
    err10_up2, err10_low2 = abs(err10_up[1]-kpi_tau),abs(err10_low[1]-kpi_tau)
    err11_up2, err11_low2 = abs(err11_up[1]-kpi_tau),abs(err11_low[1]-kpi_tau)
    err12_up2, err12_low2 = abs(err12_up[1]-kpi_tau),abs(err12_low[1]-kpi_tau)
    err13_up2, err13_low2 = abs(err13_up[1]-kpi_tau),abs(err13_low[1]-kpi_tau)

    upper2 = np.sqrt(err1_up2**2 + err2_up2**2 + err3_up2**2 + err4_up2**2 + err5_up2**2 + err6_up2**2 + err7_up2**2 + err9_up2**2 + err10_up2**2 + err11_up2**2 + err11_up2**2 + err12_up2**2 + err13_up2**2)
    lower2 = np.sqrt(err1_low2**2 + err2_low2**2 + err3_low2**2 + err4_low2**2 + err5_low2**2 + err6_low2**2 + err7_low2**2 + err9_low2**2 + err10_low2**2 + err11_low2**2 + err11_low2**2 + err12_low2**2 + err13_low2**2)

    return upper1, lower1, upper2, lower2

def itera_kpi(mK,mK_err,mpi,mpi_err,ml,ml_err,mtau,mtau_err,Vus,Vus_err,Vud,Vud_err,fKpi,fKpi_err,delt_kpi,delt_kpi_err,delt_tau,delt_tau_err,ms,ms_err,md,md_err,mu,mu_err,kpi_exp,kpi_exp_error,kpi_tau_exp,kpi_tau_exp_err):
    '''
        Iterate of mH,tanb space for kpi ratios
    '''
    sigma = 1.96

    kpi_exp_up,kpi_exp_down = kpi_exp+kpi_exp_error[0],kpi_exp+kpi_exp_error[1]
    kpi_tau_exp_up,kpi_tau_exp_down = kpi_tau_exp+kpi_tau_exp_err[0],kpi_tau_exp+kpi_tau_exp_err[1]

    av_t,av_k = 0.5*(kpi_tau_exp_up+kpi_tau_exp_down), 0.5*(kpi_exp_up+kpi_exp_down)
    sige_k, sige_t = sigma*(kpi_exp_up-av_k), sigma*(kpi_tau_exp_up-av_t)
    log_mH_range = np.linspace(0,3.5,350)
    log_tanb_range = np.linspace(-1,2,300)
    mH_range = 10**log_mH_range
    tanb_range = 10**log_tanb_range
    mH_loc = []
    tanb_loc = []
    for i in mH_range:
        for j in tanb_range:
            expect_kpi,expect_kpi_tau = decay_bsm(mK,mpi,ml,mtau,Vus,Vud,fKpi,delt_kpi,delt_tau,ms,md,mu,j,i)
#            print expect_kpi,expect_kpi_tau
            expect_kpi_uperr,expect_kpi_downerr,expect_kpi_tau_uperr,expect_kpi_tau_downerr = error_kpi(mK,mK_err,mpi,mpi_err,ml,ml_err,mtau,mtau_err,Vus,Vus_err,Vud,Vud_err,fKpi,fKpi_err,delt_kpi,delt_kpi_err,delt_tau,delt_tau_err,ms,ms_err,md,md_err,mu,mu_err,j,i)
            expect_kpi_up, expect_kpi_down = expect_kpi+expect_kpi_uperr, expect_kpi-expect_kpi_downerr
            expect_kpi_tau_up, expect_kpi_tau_down = expect_kpi_tau+expect_kpi_tau_uperr,expect_kpi_tau-expect_kpi_tau_downerr
            mid_k,mid_t=0.5*(expect_kpi_up+expect_kpi_down),0.5*(expect_kpi_tau_up+expect_kpi_tau_down)
            sig_k,sig_t=(expect_kpi_up-mid_k),(expect_kpi_tau_up-mid_t)
            kpi_bool = ((av_k >= mid_k and mid_k+sig_k >= av_k-sige_k) or (av_k <= mid_k and mid_k-sig_k <= av_k+sige_k)) 
            tkpi_bool = ((av_t >= mid_t and mid_t+sig_t >= av_t-sige_t) or (av_t <= mid_t and mid_t-sig_t <= av_t+sige_t)) 
            if kpi_bool and tkpi_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mH_loc = np.append(mH_loc,i_log)
                tanb_loc = np.append(tanb_loc,j_log)

    return mH_loc, tanb_loc

def itera_lepis(bpls_exp,bpls_exp_error,dpls_exp,dpls_exp_error,dspls_exp,dspls_exp_error,kpi_exp,kpi_exp_error,tkpi_exp,tkpi_exp_error,mbpls,mbpls_err,mdpls,mdpls_err,mdspls,mdspls_err,mK,mK_err,mpi,mpi_err,mtau,mtau_err,mmu,mmu_err,fbpls,fbpls_err,fdpls,fdpls_err,fdspls,fdspls_err,fKpi,fKpi_err,delt_kpi,delt_kpi_err,delt_tau,delt_tau_err,tbpls,tbpls_err,tdpls,tdpls_err,tdspls,tdspls_err,mu,mu_err,md,md_err,mc,mc_err,ms,ms_err,mb,mb_err,Vud,Vud_err,Vus,Vus_err,Vub,Vub_err,Vcd,Vcd_err,Vcs,Vcs_err):
    '''
        Iterate of mH,tanb space for everything
    '''
    bpls_exp_up,bpls_exp_down = bpls_exp+bpls_exp_error[0],bpls_exp+bpls_exp_error[1]
    dpls_exp_up,dpls_exp_down = dpls_exp+dpls_exp_error[0],dpls_exp+dpls_exp_error[1]
    dspls_exp_up,dspls_exp_down = dspls_exp+dspls_exp_error[0],dspls_exp+dspls_exp_error[1]
    kpi_exp_up,kpi_exp_down = kpi_exp+kpi_exp_error[0],kpi_exp+kpi_exp_error[1]
    tkpi_exp_up,tkpi_exp_down = tkpi_exp+tkpi_exp_error[0],tkpi_exp+tkpi_exp_error[1]
    log_mH_range = np.linspace(0,3.5,350)
    log_tanb_range = np.linspace(-1,2,300)
    mH_range = 10**log_mH_range
    tanb_range = 10**log_tanb_range
    mH_loc = []
    tanb_loc = []
    for i in mH_range:
        for j in tanb_range:
            bpls_the, dpls_the, dspls_the = bthe(mbpls,mtau,Vub,fbpls,tbpls,mu,mb,j,i),bthe(mdpls,mmu,Vcd,fdpls,tdpls,mc,md,j,i),bthe(mdspls,mtau,Vcs,fdspls,tdspls,mc,ms,j,i)
            bpls_err, dpls_err, dspls_err = error_branching(mbpls,mbpls_err,mtau,mtau_err,Vub,Vub_err,fbpls,fbpls_err,tbpls,tbpls_err,mu,mu_err,mb,mb_err,j,i),error_branching(mdpls,mdpls_err,mmu,mmu_err,Vcd,Vcd_err,fdpls,fdpls_err,tdpls,tdpls_err,mc,mc_err,md,md_err,j,i),error_branching(mdspls,mdspls_err,mtau,mtau_err,Vcs,Vcs_err,fdspls,fdspls_err,tdspls,tdspls_err,mc,mc_err,ms,ms_err,j,i)
            bpls_the_up,bpls_the_down,dpls_the_up,dpls_the_down,dspls_the_up,dspls_the_down=bpls_the+bpls_err[0],bpls_the-bpls_err[1],dpls_the+dpls_err[0],dpls_the-dpls_err[1],dspls_the+dspls_err[0],dspls_the-dspls_err[1]
            bpls_bool = ((bpls_exp >= bpls_the and bpls_the_up >= bpls_exp_down) or (bpls_exp <= bpls_the and bpls_the_down <= bpls_exp_up))
            dpls_bool = ((dpls_exp >= dpls_the and dpls_the_up >= dpls_exp_down) or (dpls_exp <= dpls_the and dpls_the_down <= dpls_exp_up))
            dspls_bool = ((dspls_exp >= dspls_the and dspls_the_up >= dspls_exp_down) or (dspls_exp <= dspls_the and dspls_the_down <= dspls_exp_up))

            kpi_the,tkpi_the = decay_bsm(mK,mpi,mmu,mtau,Vus,Vud,fKpi,delt_kpi,delt_tau,ms,md,mu,j,i)
            kpi_uperr,kpi_derr,tkpi_uperr,tkpi_derr = error_kpi(mK,mK_err,mpi,mpi_err,mmu,mmu_err,mtau,mtau_err,Vus,Vus_err,Vud,Vud_err,fKpi,fKpi_err,delt_kpi,delt_kpi_err,delt_tau,delt_tau_err,ms,ms_err,md,md_err,mu,mu_err,j,i)
            kpi_the_up,kpi_the_down,tkpi_the_up,tkpi_the_down = kpi_the+kpi_uperr,kpi_the-kpi_derr,tkpi_the+tkpi_uperr,tkpi_the-tkpi_derr
            kpi_bool = ((kpi_exp >= kpi_the and kpi_the_up >= kpi_exp_down) or (kpi_exp <= kpi_the and kpi_the_down <= kpi_exp_up))
            tkpi_bool = ((tkpi_exp >= tkpi_the and tkpi_the_up >= tkpi_exp_down) or (tkpi_exp <= tkpi_the and tkpi_the_down <= tkpi_exp_up))

            if bpls_bool and dpls_bool and dspls_bool  and kpi_bool and tkpi_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mH_loc = np.append(mH_loc,i_log)
                tanb_loc = np.append(tanb_loc,j_log)

    return mH_loc, tanb_loc

def bsemi(mc,mb,m_B,m_D,p,d,mH,tanb):
    '''
        normalised branching ratio for B -> D tau nu
    '''
#    rV = (-3/1.1)/(-3.4)
#    rS = (-6.6/1.89)/(-3.5)
#    r0 = (1.89/1.1)/17
#    rcb = 0.8/(1-(mc/mb))
#    gs = (m_B*tanb/mH)**2
#    NH = -rcb*gs.real*(1.038 + 0.076*rS) + (rcb**2)*(abs(gs)**2)*(0.186 + 0.017*rS)
#    R = (1.126+ 0.037*rV + (r0**2)*(1.544 + 0.082*rS + NH))/(10 - 0.95*rV)
    sH = -((tanb**2)/(1-(mc/mb)))*(((m_B**2)-(m_D**2))/(mH**2)) #0801.4938
    dp2 = p - 1.19
    dd = d - 0.46
    a0 = 0.2970 + 0.1286*dp2 + 0.7379*dd
    a1 = 0.1065 + 0.0546*dp2 + 0.4631*dd
    a2 = 0.0178 + 0.0010*dp2 + 0.0077*dd
    R = a0 + a1*sH.real + a2*abs(sH)**2
    return R

def error_bsemi(mc,mc_err,mb,mb_err,m_B,m_B_err,m_D,m_D_err,p,p_err,d,d_err,mH,tanb):
    '''
        error propagation for R(D)
    '''
    rds = bsemi(mc,mb,m_B,m_D,p,d,mH,tanb)
    err1_up = abs(bsemi(mc+mc_err[0],mb,m_B,m_D,p,d,mH,tanb)-rds)
    err1_low = abs(bsemi(mc+mc_err[1],mb,m_B,m_D,p,d,mH,tanb)-rds)
    err2_up = abs(bsemi(mc,mb+mb_err[0],m_B,m_D,p,d,mH,tanb)-rds)
    err2_low = abs(bsemi(mc,mb+mb_err[1],m_B,m_D,p,d,mH,tanb)-rds)
    err3_up = abs(bsemi(mc,mb,m_B+m_B_err[0],m_D,p,d,mH,tanb)-rds)
    err3_low = abs(bsemi(mc,mb,m_B+m_B_err[1],m_D,p,d,mH,tanb)-rds)
    err4_up = abs(bsemi(mc,mb,m_B,m_D+m_D_err[0],p,d,mH,tanb)-rds)
    err4_low = abs(bsemi(mc,mb,m_B,m_D+m_D_err[1],p,d,mH,tanb)-rds)
    err5_up = abs(bsemi(mc,mb,m_B,m_D,p+p_err[0],d,mH,tanb)-rds)
    err5_low = abs(bsemi(mc,mb,m_B,m_D,p+p_err[1],d,mH,tanb)-rds)
    err6_up = abs(bsemi(mc,mb,m_B,m_D,p,d+d_err[0],mH,tanb)-rds)
    err6_low = abs(bsemi(mc,mb,m_B,m_D,p,d+d_err[1],mH,tanb)-rds)

    upper = np.sqrt(err1_up**2 + err2_up**2 + err3_up**2 + err4_up**2 + err5_up**2 + err6_up**2)
    lower = np.sqrt(err1_low**2 + err2_low**2 + err3_low**2 + err4_low**2 + err5_low**2 + err6_low**2)

    return upper, lower

def itera_rd(mc,mc_err,mb,mb_err,m_B,m_B_err,m_D,m_D_err,p,p_err,d,d_err,rd_exp,rd_exp_err):
    '''
        Iterate over mH, tanb space
    '''
    sigma=3
    exp_branch_up,exp_branch_down = rd_exp+rd_exp_err[0],rd_exp+rd_exp_err[1]
    av_rd = 0.5*(exp_branch_up+exp_branch_down)
    sige_rd = sigma*(exp_branch_up-av_rd)
    log_mH_range = np.linspace(0,3.5,350)
    log_tanb_range = np.linspace(-1,2,300)
    mH_range = 10**log_mH_range
    tanb_range = 10**log_tanb_range
    mH_loc = []
    tanb_loc = []
    chi_rds = []
    chi_rmin = 1000
    for i in mH_range:
        for j in tanb_range:
            expect_branch = bsemi(mc,mb,m_B,m_D,p,d,i,j)
#            print expect_branch
            expect_error = error_bsemi(mc,mc_err,mb,mb_err,m_B,m_B_err,m_D,m_D_err,p,p_err,d,d_err,i,j)
            expect_branch_up, expect_branch_down = expect_branch+expect_error[0],expect_branch-expect_error[1]
            mid_rd = 0.5*(expect_branch_up+expect_branch_down)
            sig_rd = sigma*(expect_branch_up-mid_rd)
            rd_bool = ((av_rd >= mid_rd and mid_rd+sig_rd >= av_rd-sige_rd) or (av_rd <= mid_rd and mid_rd-sig_rd <= av_rd+sige_rd)) 
            if rd_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mH_loc = np.append(mH_loc,i_log)
                tanb_loc = np.append(tanb_loc,j_log)
                chi_rij = chisq_simp([av_rd],[mid_rd],[sige_rd],[sig_rd])
                chi_rds = np.append(chi_rds,chi_rij)
                if chi_rij < chi_rmin:
                    chi_rmin = chi_rij

    return mH_loc, tanb_loc, chi_rds, chi_rmin


def bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb,Vcb,alp_EM):
    '''
        Find R[B->Xsgamma/B->Xclv]
    '''
    mu0 = 2*mW
    lmu = 2*np.log(mu0/lam_QCD)
    lmub = 2*np.log(mub/lam_QCD)
    as_mu0 = (12*np.pi/(23*lmu))*(1 - (348/529)*(np.log(lmu)/lmu))
    as_mub = (12*np.pi/(23*lmub))*(1 - (348/529)*(np.log(lmub)/lmub))
    eta = as_mu0/as_mub

    lmuM = 2*np.log(mu0/mt)
    amu = as_mu0/np.pi
    factor = 1 - (4/3 + lmuM)*amu - (9.125 + 419*lmuM/72 + (2/9)*lmuM**2)*amu**2 - (0.3125*lmuM**3 + 4.5937*lmuM**2 + 25.3188*lmuM + 81.825)*amu**3
    mtmu = mt*factor

    xtW = (mtmu/mW)**2
    xtH = (mtmu/mH)**2

    F1_tW = (xtW**3 - 6*xtW**2 + 3*xtW + 2 + 6*xtW*np.log(xtW))/(12*(xtW-1)**4)
    F2_tW = (2*xtW**3 + 3*xtW**2 - 6*xtW + 1 - 6*np.log(xtW)*xtW**2)/(12*(xtW-1)**4)
    C_7SM = -(xtW/2)*(2*F1_tW + 3*F2_tW)
    C_8SM = -(3*xtW/2)*F1_tW
    hi_et = 0 
    for i in range(len(hi)):
        hi_et += hi[i]*(eta**(a[i]))

    F1_tH = (xtH**3 - 6*(xtH**2) + 3*xtH + 2 + 6*xtH*np.log(xtH))/(12*((xtH-1)**4))
    F2_tH = (2*xtH**3 + 3*xtH**2 - 6*xtH + 1 - 6*np.log(xtH)*xtH**2)/(12*((xtH-1)**4))
    F3_tH = (xtH**2 - 4*xtH + 3 + 2*np.log(xtH))/(2*((xtH-1)**3))
    F4_tH = (xtH**2 - 1 - 2*xtH*np.log(xtH))/(2*((xtH-1)**3))
    C_7H = -(xtH/2)*((1/(tanb**2))*((2/3)*F1_tH + F2_tH) + (2/3)*F3_tH + F4_tH)
    C_8H = -(xtH/2)*(F1_tH/(tanb**2) + F3_tH)
    Ceff_H = (eta**(16/23))*C_7H + (8/3)*(eta**(14/23) - eta**(16/23))*C_8H

    Ceff_SM = (eta**(16/23))*(C_7SM+C_7H) + (8/3)*(eta**(14/23) - eta**(16/23))*(C_8SM+C_8H) + hi_et

    A = A0*(1+ac*delt_mc+at*delt_mt+a_s*delt_as)
    B = B0*(1+bc*delt_mc+bt*delt_mt+bs*delt_as)
    PplN = (Ceff_SM + B*Ceff_H)**2 + A

    C = ((Vub/Vcb)**2)*(gamc/gamu)

    R = ((Vts*Vtb/Vcb)**2)*(6*alp_EM/(np.pi*C))*PplN

    return R

def error_gamma(mt,mt_err,mW,mW_err,mub,lam_QCD,QCD_err,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamc_err,gamu,gamu_err,Vub,Vub_err,Vts,Vts_err,Vtb,Vtb_err,Vcb,Vcb_err,alp_EM):
    '''
        Calculates errors in branching ratios, using functional method
        - all err vars are [up,low]
    '''
    gams = bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb,Vcb,alp_EM)
    err1_up = abs(bsgamma(mt+mt_err[0],mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb,Vcb,alp_EM)-gams)
    err1_low = abs(bsgamma(mt+mt_err[1],mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb,Vcb,alp_EM)-gams)
    err2_up = abs(bsgamma(mt,mW+mW_err[0],mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb,Vcb,alp_EM)-gams)
    err2_low = abs(bsgamma(mt,mW+mW_err[1],mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb,Vcb,alp_EM)-gams)
    err3_up = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc+gamc_err[0],gamu,Vub,Vts,Vtb,Vcb,alp_EM)-gams)
    err3_low = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc+gamc_err[1],gamu,Vub,Vts,Vtb,Vcb,alp_EM)-gams)
    err4_up = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts+Vts_err[0],Vtb,Vcb,alp_EM)-gams)
    err4_low = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts+Vts_err[1],Vtb,Vcb,alp_EM)-gams)
    err5_up = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb+Vtb_err[0],Vcb,alp_EM)-gams)
    err5_low = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb+Vtb_err[1],Vcb,alp_EM)-gams)
    err6_up = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb,Vcb+Vcb_err[0],alp_EM)-gams)
    err6_low = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb,Vcb+Vcb_err[1],alp_EM)-gams)
    err7_up = abs(bsgamma(mt,mW,mub,lam_QCD+QCD_err[0],hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb,Vcb,alp_EM)-gams)
    err7_low = abs(bsgamma(mt,mW,mub,lam_QCD+QCD_err[1],hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb,Vcb,alp_EM)-gams)
    err8_up = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu+gamu_err[0],Vub,Vts,Vtb,Vcb,alp_EM)-gams)
    err8_low = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu+gamu_err[1],Vub,Vts,Vtb,Vcb,alp_EM)-gams)
    err9_up = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub+Vub_err[0],Vts,Vtb,Vcb,alp_EM)-gams)
    err9_low = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub+Vub_err[1],Vts,Vtb,Vcb,alp_EM)-gams)

    upper = np.sqrt(err1_up**2 + err2_up**2 + err3_up**2 + err4_up**2 + err5_up**2 + err6_up**2 + err7_up**2 + err8_up**2 + err9_low**2)
    lower = np.sqrt(err1_low**2 + err2_low**2 + err3_low**2 + err4_low**2 + err5_low**2 + err6_low**2 + err7_low**2 + err8_low**2 + err9_low**2)

    return upper, lower

def iter_gamma(mt,mt_err,mW,mW_err,mub,lam_QCD,QCD_err,hi,a,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamc_err,gamu,gamu_err,Vub,Vub_err,Vts,Vts_err,Vtb,Vtb_err,Vcb,Vcb_err,alp_EM,branchc_exp,branchc_exp_error,branchs_exp,branchs_exp_error):
    '''
        Iterate over mH, tanb space
    '''
    sigma = 1.96
    branch_exp = branchs_exp/branchc_exp
    x = branch_exp*np.sqrt((branchc_exp_error[0]/branchc_exp)**2 + (branchs_exp_error[0]/branchs_exp)**2)
    y = branch_exp*np.sqrt((branchc_exp_error[1]/branchc_exp)**2 + (branchs_exp_error[1]/branchs_exp)**2)
    branch_exp_error = [x,-y]
    exp_branch_up,exp_branch_down = branch_exp+branch_exp_error[0],branch_exp+branch_exp_error[1]
    av_g = 0.5*(exp_branch_up+exp_branch_down)
    sige_g = sigma*(exp_branch_up-av_g)

    log_mH_range = np.linspace(0,3.5,350)
    log_tanb_range = np.linspace(-1,2,300)
    mH_range = 10**log_mH_range
    tanb_range = 10**log_tanb_range
    mH_loc = []
    tanb_loc = []
    for i in mH_range:
        for j in tanb_range:
            expect_branch = bsgamma(mt,mW,mub,lam_QCD,hi,a,i,j,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb,Vcb,alp_EM)
            print expect_branch
            expect_error = error_gamma(mt,mt_err,mW,mW_err,mub,lam_QCD,QCD_err,hi,a,i,j,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamc_err,gamu,gamu_err,Vub,Vub_err,Vts,Vts_err,Vtb,Vtb_err,Vcb,Vcb_err,alp_EM)
            expect_branch_up, expect_branch_down = expect_branch+expect_error[0],expect_branch-expect_error[1]
            mid_g = 0.5*(expect_branch_up+expect_branch_down)
            sig_g = sigma*(expect_branch_up-mid_g)
            br_bool = ((av_g >= mid_g and mid_g+sig_g >= av_g-sige_g) or (av_g <= mid_g and mid_g-sig_g <= av_g+sige_g)) 
            if br_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mH_loc = np.append(mH_loc,i_log)
                tanb_loc = np.append(tanb_loc,j_log)

    return mH_loc, tanb_loc

def bmumu(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH):
    '''
        Branching ratio of b(s/d) to mu mu
        tau in ps, f in MeV, mt in GeV
    '''
    Yt = 0.997*(mt/166)**1.55
    r = (mH/mt)**2

    bd1 = (3e-7)*(taubd/1.54)*pow((fbd/210),2)*pow((Vtd/0.008),2)*pow((mmu/mbd),2)*np.sqrt(1-((4*mmu**2)/(mbd**2)))
    bd2 = (1-((4*mmu**2)/(mbd**2)))*pow(((mbd**2)*(tanb**2)*np.log(r))/(8*(mW**2)*(r-1)),2) + pow((((mbd**2)*(tanb**2)*np.log(r))/(8*(mW**2)*(r-1)))-Yt,2) 
    bd = bd1*bd2

    bs1 = (1.1e-5)*(taubs/1.54)*pow((fbs/245),2)*pow((Vts/0.040),2)*pow((mmu/mbs),2)*np.sqrt(1-((4*mmu**2)/(mbs**2)))
    bs2 = (1-((4*mmu**2)/(mbs**2)))*pow(((mbs**2)*(tanb**2)*np.log(r))/(8*(mW**2)*(r-1)),2) + pow(((((mbs**2)*(tanb**2)*np.log(r))/(8*(mW**2)*(r-1)))-Yt),2) 
    bs = bs1*bs2

    return bd, bs

def bmumu2(mt,taubs,fbs,Vtb,Vts,mmu,mbs,mW,tanb,mH,mb,ms,wangle,higgs,v):
    zu = 1/tanb
    zd = -tanb
    zl = -tanb

    M = 1000
    mHh = 500
    xHp = (mH/mW)**2
    xt = (mt/mW)**2
    xb = (mb/mW)**2
    xl = (mmu/mW)**2
    xh = (higgs/mW)**2
    xHh = (mHh/mW)**2

    f1 = (-xHp+xt+xHp*np.log(xHp)-xt*np.log(xt))/(2*(xHp-xt))
    f2 = (xt - xHp*xt*np.log(xHp/xt)/(xHp-xt))/(2*(xHp-xt))
    f3 = (xHp - (xHp**2)*np.log(xHp)/(xHp-xt) + xt*(2*xHp - xt)*np.log(xt)/(xHp-xt))/(2*(xHp-xt))
    f4 = (xt*(3*xHp-xt)/2 - (xHp**2)*xt*np.log(xHp/xt)/(xHp-xt))/(4*(xHp-xt)**2)
    f5 = (xt*(xHp-3*xt)/2 - xHp*xt*(xHp-2*xt)*np.log(xHp/xt)/(xHp-xt))/(4*(xHp-xt)**2)
    f6 = (xt*(xt**2 - 3*xHp*xt + 9*xHp - 5*xt - 2)/(4*(xt-1)**2) + xHp*(xHp*xt - 3*xHp + 2*xt)*np.log(xHp)/(2*(xHp-1)*(xHp-xt)) + ((xHp**2)*(-2*xt**3 + 6*xt**2 - 9*xt + 2) + 3*xHp*(xt**2)*(xt**2 - 2*xt + 3) - (xt**2)*(2*xt**3 - 3*xt**2 + 3*xt + 1))*np.log(xt)/(2*(xHp-xt)*(xt-1)**3))/(2*(xHp-xt))
    f7 = ((xt**2 + xt - 8)*(xHp-xt)/(4*(xt-1)**2) - xHp*(xHp+2)*np.log(xHp)/(2*(xHp-1)) + (xHp*(xt**3 - 3*xt**2 + 3*xt + 2) + 3*(xt-2)*xt**2)*np.log(xt)/(2*(xt-1)**3))/(2*(xHp-xt))

    g0 = (1/(4*(xHp-xt)))*(zd*np.conj(zu)*((xt/(xHp-xt))*np.log(xHp/xt)-1) + (abs(zu)**2)*(((xt**2)/(2*(xHp-xt)**2))*np.log(xHp/xt) + (xHp-3*xt)/(4*(xHp-xt))))
    g1 = -3/4 + zd*np.conj(zu)*(xt/(xHp-xt))*(1 - (xHp/(xHp-xt))*np.log(xHp/xt)) + (abs(zu)**2)*(xt/(2*(xHp-xt)**2))*((xHp+xt)/2 - (xHp*xt/(xHp-xt))*np.log(xHp/xt))
    g2 = zd*(zd*np.conj(zu)+1)*f1 + zd*(np.conj(zu)**2)*f2 + zd*(abs(zu)**2)*f3 + zu*(abs(zu)**2)*f4 - np.conj(zu)*(abs(zu)**2)*f5 - zu*f6 - np.conj(zu)*f7

    lamh = -((higgs**2)*(-6*zu + 2*zu) - 16*mH**2 + 8*zu*M**2)/(4*zu*v**2)
    lamHh = -((mHh**2)*(6*tanb + 2*tanb) + 16*(zu**2)*mH**2 - 8*tanb*M**2)/(4*zu*v**2)

    C10Z = (abs(zu)**2)*((xt**2)/(8*wangle))*(1/(xHp-xt) - xHp*np.log(xHp/xt)/((xHp-xt)**2))
    CPZ1 = np.conj(zu)*zd*np.sqrt(xb*xl)*(xt/(16*wangle))*((xt-3*xHp)/pow(xHp-xt,2) + 2*pow(xHp,2)*np.log(xHp/xt)/pow(xHp-xt,3))
    CPZ2 = (abs(zu)**2)*np.sqrt(xb*xl)*(xt/216)*((38*xHp**2 + 54*xt*xHp**2 - 79*xHp*xt - 108*xHp*xt**2 + 47*xt**2 + 54*xt**3)/pow(xHp-xt,3) - (6*np.log(xHp/xt)*(4*xHp**3 + 9*xt*xHp**3 - 6*xt*xHp**2 - 18*(xHp**2)*(xt**2) + 9*xHp*xt**3 + 3*xt**3)/pow(xHp-xt,4)) - (3/(2*wangle))*((2*xHp**2 + 36*xt*xHp**2 - 7*xHp*xt - 72*xHp*xt**2 + 11*xt**2 + 36*xt**3)/pow(xHp-xt,3) - (6*xt*np.log(xHp/xt)*(6*xHp**3 - 12*xt*xHp**2 + 6*xHp*xt**2 + xt**2)/pow(xHp-xt,4))))
    CSbox = (np.sqrt(xl*xb)*xt/(8*(xHp-xt)*wangle))*(np.conj(zu)*zl*((xt*np.log(xt)/(xt-1)) - (xHp*np.log(xHp)/(xHp-1))) + zu*np.conj(zl)*(1 - (xHp-xt**2)*np.log(xt)/((xHp-xt)*(xt-1)) - (xHp*(xt-1)*np.log(xHp))/((xHp-xt)*(xHp-1))) + 2*zd*np.conj(zl)*np.log(xt/xHp))
    CPbox = CSbox
    CPA = -(np.sqrt(xl*xb)*zl*xt/(wangle*2*xA))*(((zu**3)*xt/2)*(1/(xHp-xt) - xHp*np.log(xHp/xt)/pow(xHp-xt,2)) + (zu/4)*(-(3*xHp*xt - 6*xHp - 2*xt**2 + 5*xt)/((xt-1)*(xHp-xt)) + xHp*(xHp**2 - 7*xHp + 6*xt)*np.log(xHp)/(pow(xHp-xt,2)*(xHp-1)) - ((xHp**2)*(xt**2 - 2*xt + 4) + 3*(xt**2)*(2*xt - 2*xHp - 1))*np.log(xt)/pow((xHp-xt)*(xt-1),2)))
    CSh = (np.sqrt(xl*xb)*xt/(wangle*2*xh))*(2*zd + 2*zu*zl)*(-2*g1*tanb + 2*g2*zu - g0**2*(v**2)*lamh/(mW**2))
    CSHh = (np.sqrt(xl*xb)*xt/(wangle*2*xHh))*(2*zu + 2*tanb*zl)*(g1*2*zu + 2*g2*tanb - g0*2*(v**2)*lamHh/(mW**2))

    C10 = -4.103 + C10Z
    C10P = 0
    CP = CPbox + CPZ1 + CPZ2 + CPA
    CPP = 0
    CS = CSbox + CSh + CSHh
    CSP = 0
    beta = np.sqrt(1-4*pow(mmu/mbs,2))
    pref = taubs*((pow(alpha,2)*g_gev*mbs*beta)/(16*np.pi**3))*pow(Vtb*Vts,2)*pow(fbs*mmu,2)
    bs1 = abs(C10-C10P+(pow(mbs,2)*(CP-CPP)/(2*mmu*(mb+ms))),2)**2
    bs2 = pow(abs(CS-CSP),2)*(pow(mbs,2)*(pow(mbs,2)-4*pow(mmu,2))/(4*pow(mmu,2)*(mb+ms)**2))
    bsi = pref*(bs1+bs2)
    bs = bsi/(1-0.061)

    return bs

def error_bmumu(mt,mt_err,taubd,taubd_err,taubs,taubs_err,fbd,fbd_err,fbs,fbs_err,Vtd,Vtd_err,Vts,Vts_err,mmu,mmu_err,mbd,mbd_err,mbs,mbs_err,mW,mW_err,tanb,mH):
    '''
        Error propagation for b(s/d) to mumu
    '''
    bd, bs = bmumu(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH)

    ## errors
    err1_up = bmumu(mt+mt_err[0],taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH)
    err1_low = bmumu(mt+mt_err[1],taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH)
    err2_up = bmumu(mt,taubd+taubd_err[0],taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH)
    err2_low = bmumu(mt,taubd+taubd_err[1],taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH)
    err3_up = bmumu(mt,taubd,taubs+taubs_err[0],fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH)
    err3_low = bmumu(mt,taubd,taubs+taubs_err[1],fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH)
    err4_up = bmumu(mt,taubd,taubs,fbd+fbd_err[0],fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH)
    err4_low = bmumu(mt,taubd,taubs,fbd+fbd_err[1],fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH)
    err5_up = bmumu(mt,taubd,taubs,fbd,fbs+fbs_err[0],Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH)
    err5_low = bmumu(mt,taubd,taubs,fbd,fbs+fbs_err[1],Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH)
    err6_up = bmumu(mt,taubd,taubs,fbd,fbs,Vtd+Vtd_err[0],Vts,mmu,mbd,mbs,mW,tanb,mH)
    err6_low = bmumu(mt,taubd,taubs,fbd,fbs,Vtd+Vtd_err[1],Vts,mmu,mbd,mbs,mW,tanb,mH)
    err7_up = bmumu(mt,taubd,taubs,fbd,fbs,Vtd,Vts+Vtd_err[0],mmu,mbd,mbs,mW,tanb,mH)
    err7_low = bmumu(mt,taubd,taubs,fbd,fbs,Vtd,Vts+Vts_err[1],mmu,mbd,mbs,mW,tanb,mH)
    err8_up = bmumu(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu+mmu_err[0],mbd,mbs,mW,tanb,mH)
    err8_low = bmumu(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu+mmu_err[1],mbd,mbs,mW,tanb,mH)
    err9_up = bmumu(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd+mbd_err[0],mbs,mW,tanb,mH)
    err9_low = bmumu(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd+mbd_err[1],mbs,mW,tanb,mH)
    err10_up = bmumu(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs+mbs_err[0],mW,tanb,mH)
    err10_low = bmumu(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs+mbs_err[1],mW,tanb,mH)
    err11_up = bmumu(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW+mW_err[0],tanb,mH)
    err11_low = bmumu(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW+mW_err[1],tanb,mH)

    ## bd
    err1_up1, err1_low1 = abs(err1_up[0]-bd),abs(err1_low[0]-bd)
    err2_up1, err2_low1 = abs(err2_up[0]-bd),abs(err2_low[0]-bd)
    err3_up1, err3_low1 = abs(err3_up[0]-bd),abs(err3_low[0]-bd)
    err4_up1, err4_low1 = abs(err4_up[0]-bd),abs(err4_low[0]-bd)
    err5_up1, err5_low1 = abs(err5_up[0]-bd),abs(err5_low[0]-bd)
    err6_up1, err6_low1 = abs(err6_up[0]-bd),abs(err6_low[0]-bd)
    err7_up1, err7_low1 = abs(err7_up[0]-bd),abs(err7_low[0]-bd)
    err8_up1, err8_low1 = abs(err8_up[0]-bd),abs(err8_low[0]-bd)
    err9_up1, err9_low1 = abs(err9_up[0]-bd),abs(err9_low[0]-bd)
    err10_up1, err10_low1 = abs(err10_up[0]-bd),abs(err10_low[0]-bd)
    err11_up1, err11_low1 = abs(err11_up[0]-bd),abs(err11_low[0]-bd)

    upper1 = np.sqrt(err1_up1**2 + err2_up1**2 + err3_up1**2 + err4_up1**2 + err5_up1**2 + err6_up1**2 + err7_up1**2 + err9_up1**2 + err10_up1**2 + err11_up1**2 + err11_up1**2)
    lower1 = np.sqrt(err1_low1**2 + err2_low1**2 + err3_low1**2 + err4_low1**2 + err5_low1**2 + err6_low1**2 + err7_low1**2 + err9_low1**2 + err10_low1**2 + err11_low1**2 + err11_low1**2)

    ## bs
    err1_up2, err1_low2 = abs(err1_up[1]-bs),abs(err1_low[1]-bs)
    err2_up2, err2_low2 = abs(err2_up[1]-bs),abs(err2_low[1]-bs)
    err3_up2, err3_low2 = abs(err3_up[1]-bs),abs(err3_low[1]-bs)
    err4_up2, err4_low2 = abs(err4_up[1]-bs),abs(err4_low[1]-bs)
    err5_up2, err5_low2 = abs(err5_up[1]-bs),abs(err5_low[1]-bs)
    err6_up2, err6_low2 = abs(err6_up[1]-bs),abs(err6_low[1]-bs)
    err7_up2, err7_low2 = abs(err7_up[1]-bs),abs(err7_low[1]-bs)
    err8_up2, err8_low2 = abs(err8_up[1]-bs),abs(err8_low[1]-bs)
    err9_up2, err9_low2 = abs(err9_up[1]-bs),abs(err9_low[1]-bs)
    err10_up2, err10_low2 = abs(err10_up[1]-bs),abs(err10_low[1]-bs)
    err11_up2, err11_low2 = abs(err11_up[1]-bs),abs(err11_low[1]-bs)

    upper2 = np.sqrt(err1_up2**2 + err2_up2**2 + err3_up2**2 + err4_up2**2 + err5_up2**2 + err6_up2**2 + err7_up2**2 + err9_up2**2 + err10_up2**2 + err11_up2**2 + err11_up2**2)
    lower2 = np.sqrt(err1_low2**2 + err2_low2**2 + err3_low2**2 + err4_low2**2 + err5_low2**2 + err6_low2**2 + err7_low2**2 + err9_low2**2 + err10_low2**2 + err11_low2**2 + err11_low2**2)

    return upper1, lower1, upper2, lower2

def error_bmumu2(mt,mt_err,taubd,taubd_err,taubs,taubs_err,fbd,fbd_err,fbs,fbs_err,Vtd,Vtd_err,Vts,Vts_err,mmu,mmu_err,mbd,mbd_err,mbs,mbs_err,mW,mW_err,tanb,mH,mb,mb_err,ms,ms_err,wangle,wan_err,higgs,higgs_err,v,v_err):
#def bmumu2(mt,taubs,fbs,Vtb,Vts,mmu,mbs,mW,tanb,mH,mb,ms,wangle,higgs,v):
    '''
        Error propagation for b(s/d) to mumu
    '''
#    bd, bs = bmumu(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH)
    bs = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH,mb,ms,wangle,higgs,v)

    ## errors
    err1_up = bmumu2(mt+mt_err[0],taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH,mb,ms,wangle,higgs,v)
    err1_low = bmumu2(mt+mt_err[1],taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH,mb,ms,wangle,higgs,v)
    err2_up = bmumu2(mt,taubd+taubd_err[0],taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH,mb,ms,wangle,higgs,v)
    err2_low = bmumu2(mt,taubd+taubd_err[1],taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH,mb,ms,wangle,higgs,v)
    err3_up = bmumu2(mt,taubd,taubs+taubs_err[0],fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH,mb,ms,wangle,higgs,v)
    err3_low = bmumu2(mt,taubd,taubs+taubs_err[1],fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH,mb,ms,wangle,higgs,v)
    err4_up = bmumu2(mt,taubd,taubs,fbd+fbd_err[0],fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH,mb,ms,wangle,higgs,v)
    err4_low = bmumu2(mt,taubd,taubs,fbd+fbd_err[1],fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH,mb,ms,wangle,higgs,v)
    err5_up = bmumu2(mt,taubd,taubs,fbd,fbs+fbs_err[0],Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH,mb,ms,wangle,higgs,v)
    err5_low = bmumu2(mt,taubd,taubs,fbd,fbs+fbs_err[1],Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH,mb,ms,wangle,higgs,v)
    err6_up = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd+Vtd_err[0],Vts,mmu,mbd,mbs,mW,tanb,mH,mb,ms,wangle,higgs,v)
    err6_low = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd+Vtd_err[1],Vts,mmu,mbd,mbs,mW,tanb,mH,mb,ms,wangle,higgs,v)
    err7_up = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd,Vts+Vtd_err[0],mmu,mbd,mbs,mW,tanb,mH,mb,ms,wangle,higgs,v)
    err7_low = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd,Vts+Vts_err[1],mmu,mbd,mbs,mW,tanb,mH,mb,ms,wangle,higgs,v)
    err8_up = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu+mmu_err[0],mbd,mbs,mW,tanb,mH,mb,ms,wangle,higgs,v)
    err8_low = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu+mmu_err[1],mbd,mbs,mW,tanb,mH,mb,ms,wangle,higgs,v)
    err9_up = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd+mbd_err[0],mbs,mW,tanb,mH,mb,ms,wangle,higgs,v)
    err9_low = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd+mbd_err[1],mbs,mW,tanb,mH,mb,ms,wangle,higgs,v)
    err10_up = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs+mbs_err[0],mW,tanb,mH,mb,ms,wangle,higgs,v)
    err10_low = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs+mbs_err[1],mW,tanb,mH,mb,ms,wangle,higgs,v)
    err11_up = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW+mW_err[0],tanb,mH,mb,ms,wangle,higgs,v)
    err11_low = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW+mW_err[1],tanb,mH,mb,ms,wangle,higgs,v)
    err12_up = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH,mb+mb_err[0],ms,wangle,higgs,v)
    err12_low = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH,mb+mb_err[1],ms,wangle,higgs,v)
    err13_up = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH,mb,ms+ms_err[0],wangle,higgs,v)
    err13_low = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH,mb,ms+ms_err[1],wangle,higgs,v)
    err14_up = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH,mb,ms,wangle+wan_err[0],higgs,v)
    err14_low = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH,mb,ms,wangle+wan_err[1],higgs,v)
    err15_up = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH,mb,ms,wangle,higgs+higgs_err[0],v)
    err15_low = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH,mb,ms,wangle,higgs+higgs_err[1],v)
    err16_up = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH,mb,ms,wangle,higgs,v+v_err[0])
    err16_low = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,tanb,mH,mb,ms,wangle,higgs,v+v_err[1])

    ## bd
    err1_up1, err1_low1 = abs(err1_up-bs),abs(err1_low-bs)
    err2_up1, err2_low1 = abs(err2_up-bs),abs(err2_low-bs)
    err3_up1, err3_low1 = abs(err3_up-bs),abs(err3_low-bs)
    err4_up1, err4_low1 = abs(err4_up-bs),abs(err4_low-bs)
    err5_up1, err5_low1 = abs(err5_up-bs),abs(err5_low-bs)
    err6_up1, err6_low1 = abs(err6_up-bs),abs(err6_low-bs)
    err7_up1, err7_low1 = abs(err7_up-bs),abs(err7_low-bs)
    err8_up1, err8_low1 = abs(err8_up-bs),abs(err8_low-bs)
    err9_up1, err9_low1 = abs(err9_up-bs),abs(err9_low-bs)
    err10_up1, err10_low1 = abs(err10_up-bs),abs(err10_low-bs)
    err11_up1, err11_low1 = abs(err11_up-bs),abs(err11_low-bs)
    err12_up1, err12_low1 = abs(err12_up-bs),abs(err12_low-bs)
    err13_up1, err13_low1 = abs(err13_up-bs),abs(err13_low-bs)
    err14_up1, err14_low1 = abs(err14_up-bs),abs(err14_low-bs)
    err15_up1, err15_low1 = abs(err15_up-bs),abs(err15_low-bs)
    err16_up1, err16_low1 = abs(err16_up-bs),abs(err16_low-bs)

    upper1 = np.sqrt(err1_up1**2 + err2_up1**2 + err3_up1**2 + err4_up1**2 + err5_up1**2 + err6_up1**2 + err7_up1**2 + err9_up1**2 + err10_up1**2 + err11_up1**2 + err11_up1**2 + err12_up1**2 + err13_up1**2 + err14_up1**2 + err15_up1**2 + err16_up1**2)
    lower1 = np.sqrt(err1_low1**2 + err2_low1**2 + err3_low1**2 + err4_low1**2 + err5_low1**2 + err6_low1**2 + err7_low1**2 + err9_low1**2 + err10_low1**2 + err11_low1**2 + err11_low1**2 + err12_low1**2 + err13_low1**2 + err14_low1**2 + err15_low1**2 + err16_low1**2)

    ## bs
#    err1_up2, err1_low2 = abs(err1_up[1]-bs),abs(err1_low[1]-bs)
#    err2_up2, err2_low2 = abs(err2_up[1]-bs),abs(err2_low[1]-bs)
#    err3_up2, err3_low2 = abs(err3_up[1]-bs),abs(err3_low[1]-bs)
#    err4_up2, err4_low2 = abs(err4_up[1]-bs),abs(err4_low[1]-bs)
#    err5_up2, err5_low2 = abs(err5_up[1]-bs),abs(err5_low[1]-bs)
#    err6_up2, err6_low2 = abs(err6_up[1]-bs),abs(err6_low[1]-bs)
#    err7_up2, err7_low2 = abs(err7_up[1]-bs),abs(err7_low[1]-bs)
#    err8_up2, err8_low2 = abs(err8_up[1]-bs),abs(err8_low[1]-bs)
#    err9_up2, err9_low2 = abs(err9_up[1]-bs),abs(err9_low[1]-bs)
#    err10_up2, err10_low2 = abs(err10_up[1]-bs),abs(err10_low[1]-bs)
#    err11_up2, err11_low2 = abs(err11_up[1]-bs),abs(err11_low[1]-bs)
#
#    upper2 = np.sqrt(err1_up2**2 + err2_up2**2 + err3_up2**2 + err4_up2**2 + err5_up2**2 + err6_up2**2 + err7_up2**2 + err9_up2**2 + err10_up2**2 + err11_up2**2 + err11_up2**2)
#    lower2 = np.sqrt(err1_low2**2 + err2_low2**2 + err3_low2**2 + err4_low2**2 + err5_low2**2 + err6_low2**2 + err7_low2**2 + err9_low2**2 + err10_low2**2 + err11_low2**2 + err11_low2**2)

    return upper1, lower1#, upper2, lower2

#def itera_bmumu(mt,mt_err,taubd,taubd_err,taubs,taubs_err,fbd,fbd_err,fbs,fbs_err,Vtd,Vtd_err,Vts,Vts_err,mmu,mmu_err,mbd,mbd_err,mbs,mbs_err,mW,mW_err,bs_exp,bs_exp_error,bd_exp,bd_exp_error):
def itera_bmumu(mt,mt_err,taubd,taubd_err,taubs,taubs_err,fbd,fbd_err,fbs,fbs_err,Vtd,Vtd_err,Vts,Vts_err,mmu,mmu_err,mbd,mbd_err,mbs,mbs_err,mW,mW_err,bs_exp,bs_exp_error,mb,mb_err,ms,ms_err,wangle,wan_err,higgs,higgs_err,v,v_err):
    '''
        Iterate of mH,tanb space for b(s/d) to mumu
    '''
    bs_exp_up,bs_exp_down = bs_exp+bs_exp_error[0],bs_exp+bs_exp_error[1]
#    bd_exp_up,bd_exp_down = bd_exp+bd_exp_error[0],bd_exp+bd_exp_error[1]
    log_mH_range = np.linspace(0,3.5,350)
    log_tanb_range = np.linspace(-1,2,300)
    mH_range = 10**log_mH_range
    tanb_range = 10**log_tanb_range
    mH_loc = []
    tanb_loc = []
    for i in mH_range:
        for j in tanb_range:
            #expect_bd,
            expect_bs = bmumu2(mt,taubd,taubs,fbd,fbs,Vtd,Vts,mmu,mbd,mbs,mW,j,i)
            print expect_bs
            #expect_bd_uperr,expect_bd_downerr,
            expect_bs_uperr,expect_bs_downerr = error_bmumu(mt,mt_err,taubd,taubd_err,taubs,taubs_err,fbd,fbd_err,fbs,fbs_err,Vtd,Vtd_err,Vts,Vts_err,mmu,mmu_err,mbd,mbd_err,mbs,mbs_err,mW,mW_err,j,i,mb,mb_err,ms,ms_err,wangle,wan_err,higgs,higgs_err,v,v_err)
#            expect_bd_up, expect_bd_down = expect_bd+expect_bd_uperr, expect_bd-expect_bd_downerr
            expect_bs_up, expect_bs_down = expect_bs+expect_bs_uperr, expect_bs-expect_bs_downerr
            #if ((bd_exp >= expect_bd and expect_bd_up >= bd_exp_down) or (bd_exp <= expect_bd and expect_bd_down <= bd_exp_up)) and ((bs_exp >= expect_bs and expect_bs_up >= bs_exp_down) or (bs_exp <= expect_bs and expect_bs_down <= bs_exp_up)):
            if ((bs_exp >= expect_bs and expect_bs_up >= bs_exp_down) or (bs_exp <= expect_bs and expect_bs_down <= bs_exp_up)):
                i_log, j_log = np.log10(i), np.log10(j)
                mH_loc = np.append(mH_loc,i_log)
                tanb_loc = np.append(tanb_loc,j_log)

    return mH_loc, tanb_loc

def itera_firstglobal(bpls_exp,bpls_exp_error,dpls_exp,dpls_exp_error,dspls_exp,dspls_exp_error,bmix_exp,bmix_exp_error,bmix_sm,bmix_sm_error,kpi_exp,kpi_exp_error,tkpi_exp,tkpi_exp_error,gams_exp,gams_exp_error,gamc_exp,gamc_exp_error,mbpls,mbpls_err,mdpls,mdpls_err,mdspls,mdspls_err,mK,mK_err,mpi,mpi_err,mB,mB_err,mtau,mtau_err,mmu,mmu_err,etaB,etaB_err,fBd,fBd_err,Bbd,Bbd_err,fbpls,fbpls_err,fdpls,fdpls_err,fdspls,fdspls_err,fKpi,fKpi_err,delt_kpi,delt_kpi_err,delt_tau,delt_tau_err,tbpls,tbpls_err,tdpls,tdpls_err,tdspls,tdspls_err,mu,mu_err,md,md_err,mc,mc_err,ms,ms_err,mb,mb_err,mt,mt_err,mtb,mtb_err,mW,mW_err,mWb,mWb_err,mub,lam_QCD,QCD_err,hi,a,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamu,gamu_err,alp_EM,Vud,Vud_err,Vus,Vus_err,Vub,Vub_err,Vcd,Vcd_err,Vcs,Vcs_err,Vcb,Vcb_err,Vtd,Vtd_err,Vts,Vts_err,Vtb,Vtb_err):
    '''
        Iterate of mH,tanb space for everything
    '''
    bpls_exp_up,bpls_exp_down = bpls_exp+bpls_exp_error[0],bpls_exp+bpls_exp_error[1]
    dpls_exp_up,dpls_exp_down = dpls_exp+dpls_exp_error[0],dpls_exp+dpls_exp_error[1]
    dspls_exp_up,dspls_exp_down = dspls_exp+dspls_exp_error[0],dspls_exp+dspls_exp_error[1]
    bmix_exp_up,bmix_exp_down = bmix_exp+bmix_exp_error[0],bmix_exp+bmix_exp_error[1]
    kpi_exp_up,kpi_exp_down = kpi_exp+kpi_exp_error[0],kpi_exp+kpi_exp_error[1]
    tkpi_exp_up,tkpi_exp_down = tkpi_exp+tkpi_exp_error[0],tkpi_exp+tkpi_exp_error[1]
    gam_exp = gams_exp/gamc_exp
    xgam = gam_exp*np.sqrt((gamc_exp_error[0]/gamc_exp)**2 + (gams_exp_error[0]/gams_exp)**2)
    ygam = gam_exp*np.sqrt((gamc_exp_error[1]/gamc_exp)**2 + (gams_exp_error[1]/gams_exp)**2)
    gam_exp_up,gam_exp_down = gam_exp+xgam,gam_exp-ygam
    log_mH_range = np.linspace(0,3.5,350)
    log_tanb_range = np.linspace(-1,2,300)
    mH_range = 10**log_mH_range
    tanb_range = 10**log_tanb_range
    mHl_loc = []
    tanbl_loc = []
    mHb_loc = []
    tanbb_loc = []
    mHg_loc = []
    tanbg_loc = []
    mHa_loc = []
    tanba_loc = []
    for i in mH_range:
        for j in tanb_range:
            bpls_the, dpls_the, dspls_the = bthe(mbpls,mtau,Vub,fbpls,tbpls,mu,mb,j,i),bthe(mdpls,mmu,Vcd,fdpls,tdpls,mc,md,j,i),bthe(mdspls,mtau,Vcs,fdspls,tdspls,mc,ms,j,i)
            bpls_err, dpls_err, dspls_err = error_branching(mbpls,mbpls_err,mtau,mtau_err,Vub,Vub_err,fbpls,fbpls_err,tbpls,tbpls_err,mu,mu_err,mb,mb_err,j,i),error_branching(mdpls,mdpls_err,mmu,mmu_err,Vcd,Vcd_err,fdpls,fdpls_err,tdpls,tdpls_err,mc,mc_err,md,md_err,j,i),error_branching(mdspls,mdspls_err,mtau,mtau_err,Vcs,Vcs_err,fdspls,fdspls_err,tdspls,tdspls_err,mc,mc_err,ms,ms_err,j,i)
            bpls_the_up,bpls_the_down,dpls_the_up,dpls_the_down,dspls_the_up,dspls_the_down=bpls_the+bpls_err[0],bpls_the-bpls_err[1],dpls_the+dpls_err[0],dpls_the-dpls_err[1],dspls_the+dspls_err[0],dspls_the-dspls_err[1]
            bpls_bool = ((bpls_exp >= bpls_the and bpls_the_up >= bpls_exp_down) or (bpls_exp <= bpls_the and bpls_the_down <= bpls_exp_up))
            dpls_bool = ((dpls_exp >= dpls_the and dpls_the_up >= dpls_exp_down) or (dpls_exp <= dpls_the and dpls_the_down <= dpls_exp_up))
            dspls_bool = ((dspls_exp >= dspls_the and dspls_the_up >= dspls_exp_down) or (dspls_exp <= dspls_the and dspls_the_down <= dspls_exp_up))

            bmix_the = mixing(mtb,i*1e3,mWb,j,Vtd,Vtb,etaB,mB,fBd,Bbd,bmix_sm)
            bmix_err = error_mixing(mtb,mtb_err,i*1e3,mWb,mWb_err,j,Vtd,Vtd_err,Vtb,Vtb_err,etaB,etaB_err,mB,mB_err,fBd,fBd_err,Bbd,Bbd_err,bmix_sm,bmix_sm_error)
            bmix_the_up, bmix_the_down = bmix_the+bmix_err[0],bmix_the-bmix_err[1]
            bmix_bool = ((bmix_exp >= bmix_the and bmix_the_up >= bmix_exp_down) or (bmix_exp <= bmix_the and bmix_the_down <= bmix_exp_up))

            kpi_the,tkpi_the = decay_bsm(mK,mpi,mmu,mtau,Vus,Vud,fKpi,delt_kpi,delt_tau,ms,md,mu,j,i)
            kpi_uperr,kpi_derr,tkpi_uperr,tkpi_derr = error_kpi(mK,mK_err,mpi,mpi_err,mmu,mmu_err,mtau,mtau_err,Vus,Vus_err,Vud,Vud_err,fKpi,fKpi_err,delt_kpi,delt_kpi_err,delt_tau,delt_tau_err,ms,ms_err,md,md_err,mu,mu_err,j,i)
            kpi_the_up,kpi_the_down,tkpi_the_up,tkpi_the_down = kpi_the+kpi_uperr,kpi_the-kpi_derr,tkpi_the+tkpi_uperr,tkpi_the-tkpi_derr
            kpi_bool = ((kpi_exp >= kpi_the and kpi_the_up >= kpi_exp_down) or (kpi_exp <= kpi_the and kpi_the_down <= kpi_exp_up))
            tkpi_bool = ((tkpi_exp >= tkpi_the and tkpi_the_up >= tkpi_exp_down) or (tkpi_exp <= tkpi_the and tkpi_the_down <= tkpi_exp_up))

            gam_the = bsgamma(mt,mW,mub,lam_QCD,hi,a,i,j,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc_exp,gamu,Vub,Vts,Vtb,Vcb,alp_EM)
            gam_err = error_gamma(mt,mt_err,mW,mW_err,mub,lam_QCD,QCD_err,hi,a,i,j,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc_exp,gamc_exp_error,gamu,gamu_err,Vub,Vub_err,Vts,Vts_err,Vtb,Vtb_err,Vcb,Vcb_err,alp_EM)
            gam_the_up,gam_the_down = gam_the+gam_err[0],gam_the-gam_err[1]
            gam_bool = ((gam_exp >= gam_the and gam_the_up >= gam_exp_down) or (gam_exp <= gam_the and gam_the_down <= gam_exp_up))
            if bpls_bool and dpls_bool and dspls_bool and kpi_bool and tkpi_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mHl_loc = np.append(mHl_loc,i_log)
                tanbl_loc = np.append(tanbl_loc,j_log)

            if bmix_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mHb_loc = np.append(mHb_loc,i_log)
                tanbb_loc = np.append(tanbb_loc,j_log)

            if gam_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mHg_loc = np.append(mHg_loc,i_log)
                tanbg_loc = np.append(tanbg_loc,j_log)

            if bpls_bool and dpls_bool and dspls_bool and bmix_bool and kpi_bool and tkpi_bool and gam_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mHa_loc = np.append(mHa_loc,i_log)
                tanba_loc = np.append(tanba_loc,j_log)

    return mHl_loc, tanbl_loc, mHb_loc, tanbb_loc, mHg_loc, tanbg_loc, mHa_loc, tanba_loc


def itera_global(bpls_exp,bpls_exp_error,dpls_exp,dpls_exp_error,dspls_exp,dspls_exp_error,bmix_exp,bmix_exp_error,bmix_sm,bmix_sm_error,kpi_exp,kpi_exp_error,tkpi_exp,tkpi_exp_error,gams_exp,gams_exp_error,gamc_exp,gamc_exp_error,mbpls,mbpls_err,mdpls,mdpls_err,mdspls,mdspls_err,mK,mK_err,mpi,mpi_err,mmB,mmB_err,mtau,mtau_err,mmu,mmu_err,etaB,etaB_err,fBd,fBd_err,Bbd,Bbd_err,fbpls,fbpls_err,fdpls,fdpls_err,fdspls,fdspls_err,fKpi,fKpi_err,delt_kpi,delt_kpi_err,delt_tau,delt_tau_err,tbpls,tbpls_err,tdpls,tdpls_err,tdspls,tdspls_err,mu,mu_err,md,md_err,mc,mc_err,ms,ms_err,mb,mb_err,mt,mt_err,mtb,mtb_err,mW,mW_err,mWb,mWb_err,mub,lam_QCD,QCD_err,hi,a,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamu,gamu_err,alp_EM,Vud,Vud_err,Vus,Vus_err,Vub,Vub_err,Vcd,Vcd_err,Vcs,Vcs_err,Vcb,Vcb_err,Vtd,Vtd_err,Vts,Vts_err,Vtb,Vtb_err,tbd,tbd_err,tbs,tbs_err,fBs,fBs_err,mbd,mbd_err,mbs,mbs_err,bs_e,bs_eerr,bd_e,bd_eerr,mB_s,mB_serr,BBs,BBs_err,bmixs_exp,bmixs_exp_error,bmixs_sm,bmixs_sm_error,m_Bmu,m_Bmu_err,p,p_err,d,d_err,rd_exp,rd_exp_err,delta_b,delta_d):
    '''
        Iterate of mH,tanb space for everything
    '''
    sigma = 1.96

    bpls_exp_up,bpls_exp_down = bpls_exp+bpls_exp_error[0],bpls_exp+bpls_exp_error[1]
    dpls_exp_up,dpls_exp_down = dpls_exp+dpls_exp_error[0],dpls_exp+dpls_exp_error[1]
    dspls_exp_up,dspls_exp_down = dspls_exp+dspls_exp_error[0],dspls_exp+dspls_exp_error[1]
    av_b,av_d,av_ds = 0.5*(bpls_exp_up+bpls_exp_down),0.5*(dpls_exp_up+dpls_exp_down),0.5*(dspls_exp_up+dspls_exp_down)
    sige_b,sige_d,sige_ds = sigma*(bpls_exp_up-av_b),sigma*(dpls_exp_up-av_d),sigma*(dspls_exp_up-av_ds)

    bmix_exp_up,bmix_exp_down = bmix_exp+bmix_exp_error[0],bmix_exp+bmix_exp_error[1]
    bmixs_exp_up,bmixs_exp_down = bmixs_exp+bmixs_exp_error[0],bmixs_exp+bmixs_exp_error[1]
    av_bmix,av_bmixs = 0.5*(bmix_exp_up+bmix_exp_down),0.5*(bmixs_exp_up+bmixs_exp_down)
    sige_bmix,sige_bmixs = sigma*(bmix_exp_up-av_bmix),sigma*(bmixs_exp_up-av_bmixs)

    kpi_exp_up,kpi_exp_down = kpi_exp+kpi_exp_error[0],kpi_exp+kpi_exp_error[1]
    tkpi_exp_up,tkpi_exp_down = tkpi_exp+tkpi_exp_error[0],tkpi_exp+tkpi_exp_error[1]
    av_k,av_t = 0.5*(kpi_exp_up+kpi_exp_down),0.5*(tkpi_exp_up+tkpi_exp_down)
    sige_k,sige_t = sigma*(kpi_exp_up-av_k),sigma*(tkpi_exp_up-av_t)

    gam_exp = gams_exp/gamc_exp
    xgam = gam_exp*np.sqrt((gamc_exp_error[0]/gamc_exp)**2 + (gams_exp_error[0]/gams_exp)**2)
    ygam = gam_exp*np.sqrt((gamc_exp_error[1]/gamc_exp)**2 + (gams_exp_error[1]/gams_exp)**2)
    gam_exp_up,gam_exp_down = gam_exp+xgam,gam_exp-ygam
    av_g = 0.5*(gam_exp_up+gam_exp_down)
    sige_g = sigma*(gam_exp_up-av_g)

    bs_exp_up,bs_exp_down = bs_e+bs_eerr[0],bs_e+bs_eerr[1]
    bd_exp_up,bd_exp_down = bd_e+bd_eerr[0],bd_e+bd_eerr[1]
    av_bs,av_bd = 0.5*(bs_exp_up+bs_exp_down),0.5*(bd_exp_up+bd_exp_down)
    sige_bs,sige_bd = sigma*(bs_exp_up-av_bs),sigma*(bd_exp_up-av_bd)

    rd_exp_up,rd_exp_down = rd_exp+rd_exp_err[0],rd_exp+rd_exp_err[1]
    av_rd = 0.5*(rd_exp_up+rd_exp_down)
    sige_rd = sigma*(rd_exp_up-av_rd)

    chi_ls,chi_ms,chi_gs,chi_mus,chi_1s,chi_2s,chi_rds=[],[],[],[],[],[],[]
    chi_lmin,chi_mmin,chi_gmin,chi_umin,chi_1min,chi_2min,chi_rmin = 100,100,100,100,[100,0,0],[100,0,0],100

    log_mH_range = np.linspace(0,3.5,350)
    log_tanb_range = np.linspace(-1,2,300)
    mH_range = 10**log_mH_range
    tanb_range = 10**log_tanb_range
    mHl_loc,mHb_loc,mHg_loc,mHa_loc,mHmu_loc,mHa2_loc,mHrd_loc = [],[],[],[],[],[],[]
    tanbl_loc,tanbb_loc,tanbg_loc,tanba_loc,tanbmu_loc,tanba2_loc,tanbrd_loc = [],[],[],[],[],[],[]
    for i in mH_range:
        for j in tanb_range:
            bpls_the, dpls_the, dspls_the = bthe(mbpls,mtau,Vub,fbpls,tbpls,mu,mb,j,i,delta_b),bthe(mdpls,mmu,Vcd,fdpls,tdpls,mc,md,j,i,delta_d),bthe(mdspls,mtau,Vcs,fdspls,tdspls,mc,ms,j,i,1)
            bpls_err, dpls_err, dspls_err = error_branching(mbpls,mbpls_err,mtau,mtau_err,Vub,Vub_err,fbpls,fbpls_err,tbpls,tbpls_err,mu,mu_err,mb,mb_err,j,i,delta_b),error_branching(mdpls,mdpls_err,mmu,mmu_err,Vcd,Vcd_err,fdpls,fdpls_err,tdpls,tdpls_err,mc,mc_err,md,md_err,j,i,delta_d),error_branching(mdspls,mdspls_err,mtau,mtau_err,Vcs,Vcs_err,fdspls,fdspls_err,tdspls,tdspls_err,mc,mc_err,ms,ms_err,j,i,1)
            bpls_the_up,bpls_the_down,dpls_the_up,dpls_the_down,dspls_the_up,dspls_the_down=bpls_the+bpls_err[0],bpls_the-bpls_err[1],dpls_the+dpls_err[0],dpls_the-dpls_err[1],dspls_the+dspls_err[0],dspls_the-dspls_err[1]
            mid_b,mid_d,mid_ds=0.5*(bpls_the_up+bpls_the_down),0.5*(dpls_the_up+dpls_the_down),0.5*(dspls_the_up+dspls_the_down)
            sig_b,sig_d,sig_ds=sigma*(bpls_the_up-mid_b),sigma*(dpls_the_up-mid_d),sigma*(dspls_the_up-mid_ds)
            bpls_bool = ((av_b >= mid_b and mid_b+sig_b >= av_b-sige_b) or (av_b <= mid_b and mid_b-sig_b <= av_b+sige_b)) 
            dpls_bool = ((av_d >= mid_d and mid_d+sig_d >= av_d-sige_d) or (av_d <= mid_d and mid_d-sig_d <= av_d+sige_d))
            dspls_bool = ((av_ds >= mid_ds and mid_ds+sig_ds >= av_ds-sige_ds) or (av_ds <= mid_ds and mid_ds-sig_ds <= av_ds+sige_ds))

            bmix_the,bmixs_the = mixing(mtb,i*1e3,mWb,j,Vtd,Vtb,etaB,mmB,fBd,Bbd,bmix_sm),mixing(mtb,i*1e3,mWb,j,Vts,Vtb,etaB,mB_s,fBs,BBs,bmixs_sm)
            bmix_err,bmixs_err = error_mixing(mtb,mtb_err,i*1e3,mWb,mWb_err,j,Vtd,Vtd_err,Vtb,Vtb_err,etaB,etaB_err,mmB,mmB_err,fBd,fBd_err,Bbd,Bbd_err,bmix_sm,bmix_sm_error),error_mixing(mtb,mtb_err,i*1e3,mWb,mWb_err,j,Vts,Vts_err,Vtb,Vtb_err,etaB,etaB_err,mB_s,mB_serr,fBs,fBs_err,BBs,BBs_err,bmixs_sm,bmixs_sm_error)
            bmix_the_up, bmix_the_down = bmix_the+bmix_err[0],bmix_the-bmix_err[1]
            bmixs_the_up, bmixs_the_down = bmixs_the+bmixs_err[0],bmixs_the-bmixs_err[1]
            mid_bm,mid_bms=0.5*(bmix_the_up+bmix_the_down),0.5*(bmixs_the_up+bmixs_the_down)
            sig_bm,sig_bms=sigma*(bmix_the_up-mid_bm),sigma*(bmixs_the_up-mid_bms)
            bmix_bool = ((av_bmix >= mid_bm and mid_bm+sig_bm >= av_bmix-sige_bmix) or (av_bmix <= mid_bm and mid_bm-sig_bm <= av_bmix+sige_bmix)) and ((av_bmixs >= mid_bms and mid_bms+sig_bms >= av_bmixs-sige_bmixs) or (av_bmixs <= mid_bms and mid_bms-sig_bms <= av_bmixs+sige_bmixs))

            kpi_the,tkpi_the = decay_bsm(mK,mpi,mmu,mtau,Vus,Vud,fKpi,delt_kpi,delt_tau,ms,md,mu,j,i)
            kpi_uperr,kpi_derr,tkpi_uperr,tkpi_derr = error_kpi(mK,mK_err,mpi,mpi_err,mmu,mmu_err,mtau,mtau_err,Vus,Vus_err,Vud,Vud_err,fKpi,fKpi_err,delt_kpi,delt_kpi_err,delt_tau,delt_tau_err,ms,ms_err,md,md_err,mu,mu_err,j,i)
            kpi_the_up,kpi_the_down,tkpi_the_up,tkpi_the_down = kpi_the+kpi_uperr,kpi_the-kpi_derr,tkpi_the+tkpi_uperr,tkpi_the-tkpi_derr
            mid_k,mid_t=0.5*(kpi_the_up+kpi_the_down),0.5*(tkpi_the_up+tkpi_the_down)
            sig_k,sig_t=sigma*(kpi_the_up-mid_k),sigma*(tkpi_the_up-mid_t)
            kpi_bool = ((av_k >= mid_k and mid_k+sig_k >= av_k-sige_k) or (av_k <= mid_k and mid_k-sig_k <= av_k+sige_k)) 
            tkpi_bool = ((av_t >= mid_t and mid_t+sig_t >= av_t-sige_t) or (av_t <= mid_t and mid_t-sig_t <= av_t+sige_t)) 

            gam_the = bsgamma(mt,mW,mub,lam_QCD,hi,a,i,j,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc_exp,gamu,Vub,Vts,Vtb,Vcb,alp_EM)
            gam_err = error_gamma(mt,mt_err,mW,mW_err,mub,lam_QCD,QCD_err,hi,a,i,j,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc_exp,gamc_exp_error,gamu,gamu_err,Vub,Vub_err,Vts,Vts_err,Vtb,Vtb_err,Vcb,Vcb_err,alp_EM)
            gam_the_up,gam_the_down = gam_the+gam_err[0],gam_the-gam_err[1]
            mid_g=0.5*(gam_the_up+gam_the_down)
            sig_g=sigma*(gam_the_up-mid_g)
            gam_bool = ((av_g >= mid_g and mid_g+sig_g >= av_g-sige_g) or (av_g <= mid_g and mid_g-sig_g <= av_g+sige_g)) 

            expect_bd,expect_bs = bmumu(mt,tbd,tbs,fBd,fBs,Vtd,Vts,mmu,mbd,mbs,mW,j,i)
            expect_bd_uperr,expect_bd_downerr,expect_bs_uperr,expect_bs_downerr = error_bmumu(mt,mt_err,tbd,tbd_err,tbs,tbs_err,fBd,fBd_err,fBs,fBs_err,Vtd,Vtd_err,Vts,Vts_err,mmu,mmu_err,mbd,mbd_err,mbs,mbs_err,mW,mW_err,j,i)
            expect_bd_up, expect_bd_down = expect_bd+expect_bd_uperr, expect_bd-expect_bd_downerr
            expect_bs_up, expect_bs_down = expect_bs+expect_bs_uperr, expect_bs-expect_bs_downerr
            mid_smu,mid_dmu=0.5*(expect_bs_up+expect_bs_down),0.5*(expect_bd_up+expect_bd_down)
            sig_smu,sig_dmu=sigma*(expect_bs_up-mid_smu),sigma*(expect_bd_up-mid_dmu)
            bs_bool = ((av_bs >= mid_smu and mid_smu+sig_smu >= av_bs-sige_bs) or (av_bs <= mid_smu and mid_smu-sig_smu <= av_bs+sige_bs)) 
            bd_bool = ((av_bd >= mid_dmu and mid_dmu+sig_dmu >= av_bd-sige_bd) or (av_bd <= mid_dmu and mid_dmu-sig_dmu <= av_bd+sige_bd))

            expect_rd = bsemi(mc,mb,m_Bmu,mdpls,p,d,i,j)
            expect_rd_err = error_bsemi(mc,mc_err,mb,mb_err,m_Bmu,m_Bmu_err,mdpls,mdpls_err,p,p_err,d,d_err,i,j)
            expect_rd_up,expect_rd_down=expect_rd+expect_rd_err[0],expect_rd-expect_rd_err[1]
            mid_rd = 0.5*(expect_rd_up+expect_rd_down)
            sig_rd = sigma*(expect_rd_up-mid_rd)
            rd_bool = ((av_rd >= mid_rd and mid_rd+sig_rd >= av_rd-sige_rd) or (av_rd <= mid_rd and mid_rd-sig_rd <= av_rd+sige_rd)) 

            if bpls_bool and dpls_bool and dspls_bool and kpi_bool and tkpi_bool:# and rd_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mHl_loc = np.append(mHl_loc,i_log)
                tanbl_loc = np.append(tanbl_loc,j_log)
                chi_lij = chisq_simp([av_b,av_d,av_ds,av_k,av_t],[mid_b,mid_d,mid_ds,mid_k,mid_t],[sige_b,sige_d,sige_ds,sige_k,sige_t],[sig_b,sig_d,sig_ds,sig_k,sig_t])
                #chi_lij = chisq_simp([av_b,av_d,av_ds,av_k,av_t,av_rd],[mid_b,mid_d,mid_ds,mid_k,mid_t,mid_rd],[sige_b,sige_d,sige_ds,sige_k,sige_t,sige_rd],[sig_b,sig_d,sig_ds,sig_k,sig_t,sig_rd])
                chi_ls = np.append(chi_ls,chi_lij)
                if chi_lij < chi_lmin:
                    chi_lmin = chi_lij

            if bmix_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mHb_loc = np.append(mHb_loc,i_log)
                tanbb_loc = np.append(tanbb_loc,j_log)
                chi_mij = chisq_simp([av_bmix,av_bmixs],[mid_bm,mid_bms],[sige_bmix,sige_bmixs],[sig_bm,sig_bms])
                chi_ms = np.append(chi_ms,chi_mij)
                if chi_mij < chi_mmin:
                    chi_mmin = chi_mij
               
            if gam_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mHg_loc = np.append(mHg_loc,i_log)
                tanbg_loc = np.append(tanbg_loc,j_log)
                chi_gij = chisq_simp([av_g],[mid_g],[sige_g],[sig_g])
                chi_gs = np.append(chi_gs,chi_gij)
                if chi_gij < chi_gmin:
                    chi_gmin = chi_gij

            if bpls_bool and dpls_bool and dspls_bool and bmix_bool and kpi_bool and tkpi_bool and gam_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mHa_loc = np.append(mHa_loc,i_log)
                tanba_loc = np.append(tanba_loc,j_log)
                chi_1ij = chisq_simp([av_b,av_d,av_ds,av_k,av_t,av_bmix,av_bmixs,av_g],[mid_b,mid_d,mid_ds,mid_k,mid_t,mid_bm,mid_bms,mid_g],[sige_b,sige_d,sige_ds,sige_k,sige_t,sige_bmix,sige_bmixs,sige_g],[sig_b,sig_d,sig_ds,sig_k,sig_t,sig_bm,sig_bms,sig_g])
                chi_1s = np.append(chi_1s,chi_1ij)
                if chi_1ij < chi_1min[0]:
                    chi_1min = [chi_1ij,i,j]
          
            if bs_bool and bd_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mHmu_loc = np.append(mHmu_loc,i_log)
                tanbmu_loc = np.append(tanbmu_loc,j_log)
                chi_uij = chisq_simp([av_bs,av_bd],[mid_smu,mid_dmu],[sige_bs,sige_bd],[sig_smu,sig_dmu])
                chi_mus = np.append(chi_mus,chi_uij)
                if chi_uij < chi_umin:
                    chi_umin = chi_uij

            if rd_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mHrd_loc = np.append(mHrd_loc,i_log)
                tanbrd_loc = np.append(tanbrd_loc,j_log)
                chi_rij = chisq_simp([av_rd],[mid_rd],[sige_rd],[sig_rd])
                chi_rds = np.append(chi_rds,chi_rij)
                if chi_rij < chi_rmin:
                    chi_rmin = chi_rij

            if bpls_bool and dpls_bool and dspls_bool and bmix_bool and kpi_bool and tkpi_bool and gam_bool and bs_bool and bd_bool and rd_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mHa2_loc = np.append(mHa2_loc,i_log)
                tanba2_loc = np.append(tanba2_loc,j_log)
                chi_2ij = chisq_simp([av_b,av_d,av_ds,av_k,av_t,av_bmix,av_bmixs,av_g,av_bs,av_bd,av_rd],[mid_b,mid_d,mid_ds,mid_k,mid_t,mid_bm,mid_bms,mid_g,mid_smu,mid_dmu,mid_rd],[sige_b,sige_d,sige_ds,sige_k,sige_t,sige_bmix,sige_bmixs,sige_g,sige_bs,sige_bd,sige_rd],[sig_b,sig_d,sig_ds,sig_k,sig_t,sig_bm,sig_bms,sig_g,sig_smu,sig_dmu,sig_rd])
                chi_2s = np.append(chi_2s,chi_2ij)
                if chi_2ij < chi_2min[0]:
                    chi_2min = [chi_2ij,i,j]

    return mHl_loc, tanbl_loc, mHb_loc, tanbb_loc, mHg_loc, tanbg_loc, mHa_loc, tanba_loc, mHmu_loc, tanbmu_loc, mHrd_loc, tanbrd_loc, mHa2_loc, tanba2_loc, chi_ls, chi_ms, chi_gs, chi_1s, chi_mus, chi_rds, chi_2s, chi_lmin, chi_mmin, chi_gmin, chi_1min, chi_umin, chi_rmin, chi_2min







































