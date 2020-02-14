from __future__ import division
import numpy as np
from functions import rh

g_gev = (1.1663787e-5)**2
hbar_gev = 6.582119514e-25
g_mev = (1.1663787e-11)**2
hbar_mev = 6.582119514e-22

def vsm(mm,ml,fm,taum):
    '''
        Calculates SM branching ratio
    '''
    Bs = (1/(8*np.pi))*(g_gev*mm*ml**2)*((1-(ml**2/mm**2))**2)*(fm**2)*taum
    return Bs

def vthe(mm,ml,fm,taum,mu,md,tanb,mH,exp):
    '''
        bsm*(1+rh)^2 to check against exp
    '''
    branching = vsm(mm,ml,fm,taum)*(1+rh(mu,md,mm,tanb,mH))**2
    V = exp/branching
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

def ckmel(V,V_err,mu,mu_err,md,md_err,mm,mm_err,ml,ml_err,fm,fm_err,taum,taum_err,exp,exp_err):
    '''
        an explanation
    '''
    log_mH_range = np.linspace(0,3.5,350)
    log_tanb_range = np.linspace(-1,2,300)
    mH_range = 10**log_mH_range
    tanb_range = 10**log_tanb_range 
    mH_loc,tanb_loc,val_loc = [],[],[]
    for i in mH_range:
        for j in tanb_range:
            expect_v = vthe(mm,ml,fm,taum,mu,md,j,i,exp)
            expect_err = error_vranching(mm,mm_err,ml,ml_err,fm,fm_err,taum,taum_err,mu,mu_err,md,md_err,j,i,exp,exp_err)
            expect_up,expect_down = expect_v+expect_err[0],expect_v-expect_err[1]
            ve = vs(V,mu,md,mm,j,i)
            verr = error_vs(V,V_err,mu,mu_err,md,md_err,mm,mm_err,j,i)
            ve_up,ve_d = ve+verr[0],ve-verr[1]
            if (ve >= expect_v and expect_up >= ve_d) or (ve <= expect_v and expect_down <= ve_up):
                i_log, j_log = np.log10(i), np.log10(j)
                mH_loc = np.append(mH_loc,i_log)
                tanb_loc = np.append(tanb_loc,j_log)
                val_loc = np.append(val_loc,ve)

    return mH_loc, tanb_loc, val_loc


    

