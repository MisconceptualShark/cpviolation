from __future__ import division 
import numpy as np

g_gev = (1.1663787e-5)**2
hbar_gev = 6.582119514e-25
g_mev = (1.1663787e-11)**2
hbar_mev = 6.582118514e-22

def bsm(mm,ml,Vud,fm,taum):
    '''
        Calculates SM branching ratio
    '''
    Bs = (1/(8*np.pi))*(g_gev*mm*ml**2)*((1-(ml**2/mm**2))**2)*(Vud**2)*(fm**2)*taum
    return Bs

def rh(mu,md,mm,tanb,mH):
    '''
        Returns 2HDM correction factor rh
    '''
    r = ((mu-md*tanb**2)/(mu+md))*(mm/mH)**2
    return r

def bthe(mm,ml,Vud,fm,taum,mu,md,tanb,mH):
    '''
        bsm*(1+rh)^2 to check against exp
    '''
    branching = bsm(mm,ml,Vud,fm,taum)*(1+rh(mu,md,mm,tanb,mH))**2
    return branching

def error_branching(mm,mm_err,ml,ml_err,Vud,Vud_err,fm,fm_err,taum,taum_err,mu,mu_err,md,md_err,tanb,mH):
    '''
        Calculates errors in branching ratios, using functional method
        - all err vars are [up,low]
    '''
    brt = bthe(mm,ml,Vud,fm,taum,mu,md,tanb,mH)
    err1_up = abs(bthe(mm+mm_err[0],ml,Vud,fm,taum,mu,md,tanb,mH)-brt)
    err1_low = abs(bthe(mm+mm_err[1],ml,Vud,fm,taum,mu,md,tanb,mH)-brt)
    err2_up = abs(bthe(mm,ml+ml_err[0],Vud,fm,taum,mu,md,tanb,mH)-brt)
    err2_low = abs(bthe(mm,ml+ml_err[1],Vud,fm,taum,mu,md,tanb,mH)-brt)
    err3_up = abs(bthe(mm,ml,Vud+Vud_err[0],fm,taum,mu,md,tanb,mH)-brt)
    err3_low = abs(bthe(mm,ml,Vud+Vud_err[1],fm,taum,mu,md,tanb,mH)-brt)
    err4_up = abs(bthe(mm,ml,Vud,fm+fm_err[0],taum,mu,md,tanb,mH)-brt)
    err4_low = abs(bthe(mm,ml,Vud,fm+fm_err[1],taum,mu,md,tanb,mH)-brt)
    err5_up = abs(bthe(mm,ml,Vud,fm,taum+taum_err[0],mu,md,tanb,mH)-brt)
    err5_low = abs(bthe(mm,ml,Vud,fm,taum+taum_err[1],mu,md,tanb,mH)-brt)
    err6_up = abs(bthe(mm,ml,Vud,fm,taum,mu+mu_err[0],md,tanb,mH)-brt)
    err6_low = abs(bthe(mm,ml,Vud,fm,taum,mu+mu_err[1],md,tanb,mH)-brt)
    err7_up = abs(bthe(mm,ml,Vud,fm,taum,mu,md+md_err[0],tanb,mH)-brt)
    err7_low = abs(bthe(mm,ml,Vud,fm,taum,mu,md+md_err[1],tanb,mH)-brt)

    upper = np.sqrt(err1_up**2 + err2_up**2 + err3_up**2 + err4_up**2 + err5_up**2 + err6_up**2 + err7_up**2)
    lower = np.sqrt(err1_low**2 + err2_low**2 + err3_low**2 + err4_low**2 + err5_low**2 + err6_low**2 + err7_low**2)

    return upper, lower

def itera(mm,mm_err,ml,ml_err,Vud,Vud_err,fm,fm_err,taum,taum_err,mu,mu_err,md,md_err,branch_exp,branch_exp_error):
    '''
        Choose min,max limits for scope of tan(beta) and mH+, then check for each point in this if:
            - upper error on branching sm is above the lower error on branching exp
            - lower error on branching sm is below the upper error on branching exp
        If either is true, plot a point at coordinate, and tadaa
    '''
    exp_branch_up,exp_branch_down = branch_exp+branch_exp_error[0],branch_exp+branch_exp_error[1]
    log_mH_range = np.linspace(0,3,300)
    log_tanb_range = np.linspace(-1,2,300)
    mH_range = 10**log_mH_range
    tanb_range = 10**log_tanb_range
    mH_loc = []
    tanb_loc = []
    for i in mH_range:
        for j in tanb_range:
            expect_branch = bthe(mm,ml,Vud,fm,taum,mu,md,j,i)
            print expect_branch
            expect_error = error_branching(mm,mm_err,ml,ml_err,Vud,Vud_err,fm,fm_err,taum,taum_err,mu,mu_err,md,md_err,j,i)
            expect_branch_up, expect_branch_down = expect_branch+expect_error[0],expect_branch-expect_error[1]
            if (branch_exp >= expect_branch and expect_branch_up >= exp_branch_down) or (branch_exp <= expect_branch and expect_branch_down <= exp_branch_up):
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

    return delt_mq #/hbar_mev

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
    exp_branch_up,exp_branch_down = branch_exp+branch_exp_error[0],branch_exp+branch_exp_error[1]
    log_mH_range = np.linspace(3,6,300)
    log_tanb_range = np.linspace(-1,2,300)
    mH_range = 10**log_mH_range
    tanb_range = 10**log_tanb_range
    mH_loc = []
    tanb_loc = []
    for i in mH_range:
        for j in tanb_range:
            expect_branch = mixing(mt,i,mW,j,Vtq,Vtb,etaB,mB,fBq,BBq,expect)
            print expect_branch
            expect_error = error_mixing(mt,mt_err,i,mW,mW_err,j,Vtq,Vtq_err,Vtb,Vtb_err,etaB,etaB_err,mB,mB_err,fBq,fBq_err,BBq,BBq_err,expect,expect_err)
            expect_branch_up, expect_branch_down = expect_branch+expect_error[0],expect_branch-expect_error[1]
            if (branch_exp >= expect_branch and expect_branch_up >= exp_branch_down) or (branch_exp <= expect_branch and expect_branch_down <= exp_branch_up):
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
    kpi_exp_up,kpi_exp_down = kpi_exp+kpi_exp_error[0],kpi_exp+kpi_exp_error[1]
    kpi_tau_exp_up,kpi_tau_exp_down = kpi_tau_exp+kpi_tau_exp_err[0],kpi_tau_exp+kpi_tau_exp_err[1]
    log_mH_range = np.linspace(0,3,300)
    log_tanb_range = np.linspace(-1,2,300)
    mH_range = 10**log_mH_range
    tanb_range = 10**log_tanb_range
    mH_loc = []
    tanb_loc = []
    for i in mH_range:
        for j in tanb_range:
            expect_kpi,expect_kpi_tau = decay_bsm(mK,mpi,ml,mtau,Vus,Vud,fKpi,delt_kpi,delt_tau,ms,md,mu,j,i)
            print expect_kpi,expect_kpi_tau
            expect_kpi_uperr,expect_kpi_downerr,expect_kpi_tau_uperr,expect_kpi_tau_downerr = error_kpi(mK,mK_err,mpi,mpi_err,ml,ml_err,mtau,mtau_err,Vus,Vus_err,Vud,Vud_err,fKpi,fKpi_err,delt_kpi,delt_kpi_err,delt_tau,delt_tau_err,ms,ms_err,md,md_err,mu,mu_err,j,i)
            expect_kpi_up, expect_kpi_down = expect_kpi+expect_kpi_uperr, expect_kpi-expect_kpi_downerr
            expect_kpi_tau_up, expect_kpi_tau_down = expect_kpi_tau+expect_kpi_tau_uperr,expect_kpi_tau-expect_kpi_tau_downerr
            if ((kpi_exp >= expect_kpi and expect_kpi_up >= kpi_exp_down) or (kpi_exp <= expect_kpi and expect_kpi_down <= kpi_exp_up)) and ((kpi_tau_exp >= expect_kpi_tau and expect_kpi_tau_up >= kpi_tau_exp_down) or (kpi_tau_exp <= expect_kpi_tau and expect_kpi_tau_down <= kpi_tau_exp_up)):
                i_log, j_log = np.log10(i), np.log10(j)
                mH_loc = np.append(mH_loc,i_log)
                tanb_loc = np.append(tanb_loc,j_log)

    return mH_loc, tanb_loc

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
    branch_exp = branchs_exp/branchc_exp
    x = branch_exp*np.sqrt((branchc_exp_error[0]/branchc_exp)**2 + (branchs_exp_error[0]/branchs_exp)**2)
    y = branch_exp*np.sqrt((branchc_exp_error[1]/branchc_exp)**2 + (branchs_exp_error[1]/branchs_exp)**2)
    branch_exp_error = [x,-y]
    exp_branch_up,exp_branch_down = branch_exp+branch_exp_error[0],branch_exp+branch_exp_error[1]
    log_mH_range = np.linspace(0,3,300)
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
            if (branch_exp >= expect_branch and expect_branch_up >= exp_branch_down) or (branch_exp <= expect_branch and expect_branch_down <= exp_branch_up):
                i_log, j_log = np.log10(i), np.log10(j)
                mH_loc = np.append(mH_loc,i_log)
                tanb_loc = np.append(tanb_loc,j_log)

    return mH_loc, tanb_loc




