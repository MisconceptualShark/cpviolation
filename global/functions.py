from __future__ import division 
import numpy as np
from rdstarring import *
from fitting import *
from scipy.integrate import quad

g_gev = (1.1663787e-5)**2
Gf = 1.1663787e-5
hbar_gev = 6.582119514e-25
g_mev = (1.1663787e-11)**2
hbar_mev = 6.582118514e-22

########## LEPTONICS ##########

def bsm(mm,ml,Vud,fm,taum,delta):
    '''
        Calculates SM branching ratio
    '''
    Bs = (1/(8*np.pi))*(g_gev*mm*ml**2)*((1-(ml**2/mm**2))**2)*(Vud**2)*(fm**2)*taum*delta
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

########## MIXING ##########

def mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq,lam,mbo):
    '''
        B mixing mass eqn
    '''
    def Li2(x):
        '''
        Special function
        '''
        def func(t):
            z = np.log(1-t)/t
            return z
        inte, err = quad(func,0,x)
        return -1*inte
    lmu = 2*np.log(mW/lam)
    as_m = (12*np.pi/(23*lmu))*(1 - (348/529)*(np.log(lmu)/lmu))
    lmuM = 2*np.log(mW/mt)
    amu = as_m/np.pi
    lmut = 2*np.log(mt/lam)
    as_mt = (12*np.pi/(23*lmut))*(1 - (348/529)*(np.log(lmut)/lmut))
    lmuMt = 2*np.log(mt/mt)
    amut = as_mt/np.pi
    factor1 = 1 - (4/3 + lmuM)*amu - (9.125 + 419*lmuM/72 + (2/9)*lmuM**2)*amu**2 - (0.3125*lmuM**3 + 4.5937*lmuM**2 + 25.3188*lmuM + 81.825)*amu**3
    factor2 = 1 - (4/3 + lmuMt)*amut - (9.125 + 419*lmuMt/72 + (2/9)*lmuMt**2)*amut**2 - (0.3125*lmuMt**3 + 4.5937*lmuMt**2 + 25.3188*lmuMt + 81.825)*amut**3
    mtmu = mt*factor1
    mtmut = 165.297#mt*factor2
    gam0,gam1,b0,b1 = 4,-43/9,23/3,116/3
    lmb = 2*np.log(mbo/lam)
    as_b = (12*np.pi/(23*lmb))*(1 - (348/529)*(np.log(lmb)/lmb))
    Bab = pow(as_b,-6/23)*(1-(as_b/(4*np.pi))*(gam1/(2*b0) - gam0*b1/(2*b0**2)))

    x_tH1 = (mtmut/mH)**2
    x_tW1 = (mtmut/mW)**2
    x_tH = (mtmu/mH)**2
    x_tW = (mtmu/mW)**2
    S_WW = x_tW1*(1/4 +9/(4*(1-x_tW1)) - 3/(2*(1-x_tW1)**2) - 3*(x_tW1**2)*np.log(x_tW1)/(2*(1-x_tW1)**3))
    S_WW1 = x_tW*(1/4 +9/(4*(1-x_tW)) - 3/(2*(1-x_tW)**2) -3*(x_tW**2)*np.log(x_tW)/(2*(1-x_tW)**3))
    S_WH = (x_tH1*x_tW1/(4*tanb**2))*((2*x_tW1-8*x_tH1)*np.log(x_tH1)/((x_tH1-x_tW1)*(1-x_tH1)**2) + 6*x_tW1*np.log(x_tW1)/((x_tH1-x_tW1)*(1-x_tW1)**2) - (8-2*x_tW1)/((1-x_tW1)*(1-x_tH1)))
    S_WH1 = (x_tH*x_tW/(4*tanb**2))*((2*x_tW-8*x_tH)*np.log(x_tH)/((x_tH-x_tW)*(1-x_tH)**2) + 6*x_tW*np.log(x_tW)/((x_tH-x_tW)*(1-x_tW)**2) - (8-2*x_tW)/((1-x_tW)*(1-x_tH)))
    S_HH = (x_tH1*x_tW1/(4*tanb**4))*((1+x_tH1)/((1-x_tH1)**2)+2*x_tH1*np.log(x_tH1)/((1-x_tH1)**3))
    S_HH1 = (x_tH*x_tW/(4*tanb**4))*((1+x_tH)/((1-x_tH)**2)+2*x_tH*np.log(x_tH)/((1-x_tH)**3))

#    Lo = Li2(1-1/x_tW)
#    Lu = Li2(1-x_tW)
#    WW1tt = (4*x_tW+38*(x_tW**2)+6*(x_tW**3))*np.log(x_tW)/(x_tW-1)**4 +(12*x_tW+48*(x_tW**2)+12*(x_tW**3))*Lo/(x_tW-1)**4 +(24*x_tW+48*(x_tW**2))*Lu/(x_tW-1)**4 -(3+28*x_tW+17*(x_tW**2))/(x_tW-1)**3
#    WW1tu = 2*(3+13*x_tW)/(x_tW-1)**2 - 2*x_tW*(5+11*x_tW)*np.log(x_tW)/(x_tW-1)**3 - 12*x_tW*(1+3*x_tW)*Lo/(x_tW-1)**3 - 24*x_tW*(1+x_tW)*Lu/(x_tW-1)**3
#    PP1 = -(x_tW**2)*(7+52*x_tW-11*(x_tW**2))/(4*(x_tW-1)**3) + 3*(x_tW**3)*(4+5*x_tW-x_tW**2)*np.log(x_tW)/(2*(x_tW-1)**4) +3*(x_tW**3)*(3+4*x_tW-x_tW**2)*Lo/(x_tW-1)**4 +18*(x_tW**3)*Lu/(x_tW-1)**4
#    WP1 = 4*(x_tW**2)*(11+13*x_tW)/(x_tW-1)**3  + 2*(x_tW**2)*(5+x_tW)*(1-9*x_tW)*np.log(x_tW)/(x_tW-1)**4 - 24*(x_tW**2)*(1+4*x_tW+x_tW**2)*Lo/(x_tW-1)**4 - 48*(x_tW**2)*(1+2*x_tW)*Lu/(x_tW-1)**4
#    WW1 = WW1tt - WW1tu + 3
#    L1s = WW1 + WP1 + PP1
#
#    WW8tt = 2*x_tW*(4-3*x_tW)*np.log(x_tW)/(x_tW-1)**3 - (12*x_tW-12*x_tW**2 -8*x_tW**3)*Lo/(x_tW-1)**4 + (8-12*x_tW+12*x_tW**2)*Lu/(x_tW-1)**4 - (23-x_tW)/(x_tW-1)**2 
#    WW8tu = 2*(2-x_tW)*(np.pi**2)/(3*x_tW) - (8-5*x_tW)*np.log(x_tW)/(x_tW-1)**2 - (6*x_tW+4*x_tW**2)*Lo/(x_tW*(x_tW-1)**2) + (8+12*x_tW-6*x_tW**2)*Lu/(x_tW*(x_tW-1)**2) - 15/(x_tW-1)
#    PP8 = -11*(x_tW**2)*(1+x_tW)/(4*(x_tW-1)**2) + (x_tW**3)*(4-3*x_tW)*np.log(x_tW)/(2*(x_tW-1)**3) + (x_tW**3)*(3-3*x_tW+2*x_tW**2)*Lo/(x_tW-1)**4 + (x_tW**2)*(2+3*x_tW-3*x_tW**2)*Lu/(x_tW-1)**4
#    WP8 = 30*(x_tW**2)/(x_tW-1)**2 + 12*(x_tW**3)*np.log(x_tW)/(x_tW-1)**3 - 12*(x_tW**4)*Lo/(x_tW-1)**4 - 12*(x_tW**2)*(2-x_tW**2)*Lu/(x_tW-1)**4 
#    WW8 = WW8tt - WW8tu - 23 + (4/3)*np.pi**2
#    L8s = WW8 + WP8 + PP8
#
#    Loh = Li2(1-1/x_tH)
#    Luh = Li2(1-x_tH)
#    Louh = Li2(1-x_tH/x_tW)
#    PP1h = -(x_tH**2)*(7+52*x_tH-11*(x_tH**2))/(4*(x_tH-1)**3) + 3*(x_tH**3)*(4+5*x_tH-x_tH**2)*np.log(x_tH)/(2*(x_tH-1)**4) + 3*(x_tH**3)*(3+4*x_tH-x_tH**2)*Loh/(x_tH-1)**4 + 18*(x_tH**3)*Luh/(x_tH-1)**4
#    PP8h = -11*(x_tH**2)*(1+x_tH)/(4*(x_tH-1)**2) + (x_tH**3)*(4-3*x_tH)*np.log(x_tH)/(2*(x_tH-1)**3) + (x_tH**3)*(3-3*x_tH+2*x_tH**2)*Loh/(x_tH-1)**4 + (x_tH**2)*(2+3*x_tH-3*x_tH**2)*Luh/(x_tH-1)**4
#    dSHH = (x_tW*(x_tH**2)/(4*tanb**4))*((1+x_tH)/(1-x_tH)**2 + 2*x_tH*np.log(x_tH)/(1-x_tH)**3)
#    dSHW = ((x_tW**3)/(4*tanb**4))*((1+x_tW)/(1-x_tW)**2 + 2*x_tW*np.log(x_tW)/(1-x_tW)**3)
#    HH1 = (x_tW/x_tH)*PP1h+6*(np.log(x_tH)-np.log(x_tW))*(dSHH+dSHW)
#    WH1 = x_tW*(2*(x_tH**2)*(13+3*x_tH)*np.log(x_tH)/((x_tH-x_tW)*(x_tH-1)**3) - 2*x_tH*(9+7*x_tH+7*x_tW-23*x_tW*x_tH)/pow((x_tW-1)*(x_tH-1),2) - 2*(x_tH**2)*(18-6*x_tH-44*x_tW+13*x_tH*x_tW+9*x_tH*x_tW**2)*np.log(x_tW)/((x_tH-x_tW)*(x_tH-1)**2 *(x_tW-1)**3) - 24*(x_tH**2)*np.log(x_tH)*np.log(x_tW)/((x_tH-x_tW)*(x_tH-1)**3) + 24*(x_tH**2)*Loh/((x_tH-x_tW)*(x_tH-1)**2) - 24*x_tH*x_tW*(1+x_tW)*Lo/((x_tH-x_tW)*(x_tW-1)**3) - 48*x_tW*x_tH*Lu/((x_tH-x_tW)*(x_tW-1)**3))
#    PH1 = (x_tW**2)*(x_tH*(31-15*x_tH-15*x_tW-x_tH*x_tW)/(2*pow((x_tH-1)*(x_tW-1),2)) - x_tH*(7+21*x_tH-12*x_tH**2)*np.log(x_tH)/(2*(x_tH-x_tW)*(x_tH-1)**3) + x_tH*(7-9*x_tW+36*x_tW**2 - 18*x_tW**3)*np.log(x_tW)/(2*(x_tH-x_tW)*(x_tH-1)**2 *(x_tW-1)**3) + (x_tH**2)*(8-36*x_tW+9*x_tW**2 + 3*x_tW**3)*np.log(x_tW)/((x_tH-x_tW)*(x_tH-1)**2 *(x_tW-1)**3) - (x_tH**3)*(11-45*x_tW+18*x_tW**2)*np.log(x_tW)/(2*(x_tH-x_tW)*(x_tH-1)**2 *(x_tW-1)**3) + 6*x_tH*np.log(x_tH)*np.log(x_tW)/((x_tH-x_tW)*(x_tH-1)**3) - 6*x_tH*(1+x_tH-x_tH**2)*Loh/((x_tH-x_tW)*(x_tH-1)**2) + 6*x_tH*(1+2*x_tW**2 -x_tW**3)*Lo/((x_tH-x_tW)*(x_tW-1)**3) + 12*x_tH*Lu/((x_tH-x_tW)*(x_tW-1)**3))
#    HH8 = (x_tW/x_tH)*PP8h+6*(np.log(x_tH)-np.log(x_tW))*S_HH1
#    WH8 = x_tW*(24*x_tH*x_tW*Lu/((x_tH-x_tW)*(x_tW-1)**2) + 6*(x_tH**2)*(5*x_tW-x_tH+3*x_tH*x_tW**2)*Lo/((x_tH-x_tW)*pow((x_tH-1)*(x_tW-1),2)*x_tW) + 6*x_tH*(2*x_tW**2 -10*x_tH*x_tW+x_tH*x_tW**2)*Lo/((x_tH-x_tW)*pow((x_tH-1)*(x_tW-1),2)) + 6*(x_tH**2)*(5*x_tW-x_tH-8*x_tW**2 +2*x_tH*x_tW**2)*Luh/((x_tH-x_tW)*pow((x_tH-1)*(x_tW-1),2)*x_tW) + 6*(x_tW**2 -x_tH*x_tW+2*(x_tH*x_tW)**2)*Luh/((x_tH-x_tW)*pow((x_tH-1)*(x_tW-1),2)) + 6*(x_tH**2)*(-x_tH+5*x_tW)*Loh/(x_tW*(x_tH-x_tW)*(x_tH-1)**2) - 6*(x_tH**2)*(5*x_tW-x_tH-8*x_tW**2 +2*x_tH*x_tW**2)*Louh/(x_tW*(x_tH-x_tW)*pow((x_tH-1)*(x_tW-1),2)) - 6*(x_tW**2 -x_tH*x_tW+2*(x_tH*x_tW)**2)*Louh/((x_tH-x_tW)*pow((x_tH-1)*(x_tW-1),2)) - 6*(x_tH**2)*(1-x_tH-np.log(x_tH))/((x_tW-1)*(x_tH-1)**2) + 6*x_tH*(2*x_tW-1)*np.log(x_tW)/((x_tH-1)*(x_tW-1)**2) + 6*(x_tH**2)*(5*x_tW-x_tH-8*x_tW**2)*np.log(x_tH)*np.log(x_tW)/(x_tW*(x_tH-x_tW)*pow((x_tH-1)*(x_tW-1),2)) + 12*(x_tH**2)*(x_tH*x_tW+x_tW**2)*np.log(x_tH)*np.log(x_tW)/((x_tH-x_tW)*pow((x_tH-1)*(x_tW-1),2)))
#    PH8 = (x_tW**2)*((2*x_tH+2*x_tW-11*x_tH*x_tW)/(2*x_tW*(x_tW-1)*(x_tH-1)) - (2*x_tH**2 -7*x_tH*x_tW+2*x_tW*x_tH**2 +2*x_tW**2 +x_tH*x_tW**2)*np.log(x_tH)/(2*x_tW*(x_tW-1)*(x_tH-x_tW)*(x_tH-1)**2) - x_tH*(7-7*x_tH+4*x_tW-6*x_tW**2)*np.log(x_tW)/(2*(x_tH-1)*(x_tH-x_tW)*(x_tW-1)**2) + (x_tH**2 +x_tW**2 -3*pow(x_tH*x_tW,2))*np.log(x_tW)/(x_tW*(x_tH-1)*(x_tH-x_tW)*(x_tW-1)**2) - (x_tH**2)*(4-6*x_tW+3*x_tH*x_tW)*np.log(x_tH)*np.log(x_tW)/(x_tW*(x_tH-x_tW)*pow((x_tH-1)*(x_tW-1),2)) + x_tH*(x_tH**2 -3*x_tW**2 +6*x_tW**3 -3*x_tW**4)*np.log(x_tH)*np.log(x_tW)/((x_tH-x_tW)*pow(x_tW*(x_tH-1)*(x_tW-1),2)) - x_tH*(3*x_tW**2 +2*x_tH*x_tW*(2+x_tW)-(x_tH**2)*(1+2*x_tW))*Loh/((x_tH-x_tW)*pow(x_tW*(x_tH-1),2)) - (4*x_tH*x_tW-6*x_tW*x_tH**2 +3*pow(x_tH*x_tW,2)-x_tW**2)*Luh/(x_tH*(x_tH-x_tW)*pow((x_tH-1)*(x_tW-1),2)) - (4*x_tW*x_tH**2 -6*pow(x_tH*x_tW,2)-x_tH**3 +3*(x_tW**2)*(x_tH**3))*Luh/((x_tH-x_tW)*pow(x_tW*(x_tW-1)*(x_tH-1),2)) + 2*(x_tH**2)*(6-x_tW**2 -3*x_tH+x_tW*x_tH)*Lo/((x_tH-x_tW)*pow((x_tH-1)*(x_tW-1),2)) - x_tH*(3*x_tW**2 +4*x_tH*x_tW-x_tH**2)*Lo/((x_tH-x_tW)*pow(x_tW*(x_tW-1)*(x_tH-1),2)) + (4*x_tH*x_tW-6*x_tW*x_tH**2 +3*pow(x_tH*x_tW,2)-x_tW**2)*Louh/(x_tH*(x_tH-x_tW)*pow((x_tH-1)*(x_tW-1),2)) + (x_tH**2)*(4*x_tW-6*x_tW**2 -x_tH+3*x_tH*x_tW**2)*Louh/((x_tH-x_tW)*pow(x_tW*(x_tH-1)*(x_tW-1),2)) - 6*x_tH*Lu/((x_tH-x_tW)*(x_tW-1)**2))
#    L1h = (1/tanb**2)*WH1 + (1/tanb**2)*PH1 + (1/tanb**4)*HH1
#    L8h = (1/tanb**2)*WH8 + (1/tanb**2)*PH8 + (1/tanb**4)*HH8
#
#    CA = 1/3
#    CF = 4/3
#    Dsm = CA*(L8s+S_WW1*5)+CF*(L1s+3*S_WW1)
#    DH = CF*(L1h+3*(S_WH1+S_HH1))+CA*(L8h+5*(S_WH1+S_HH1))
#    Dx = Dsm + DH
#    Sx = S_WW1+S_WH1+S_HH1
#    Z = -5165/3174
#    eta2 = pow(as_m,6/23)*Bab*(1+(as_m/(4*np.pi))*(Dx/Sx + Z))

    #change eta2 to etaB to ignore NLO and vice versa
    delt_mq = (g_gev/(6*np.pi**2))*((Vtq*Vtb)**2)*etaB*mB*(mW**2)*(fBq)*(S_WW+S_WH+S_HH)

    return delt_mq/hbar_gev

def error_mixing(mt,mt_err,mH,mW,mW_err,tanb,Vtq,Vtq_err,Vtb,Vtb_err,etaB,etaB_err,mB,mB_err,fBq,fBq_err,BBq,BBq_err,lam,lam_err,m_b,m_b_err):
    '''
        Calculates errors in branching ratios, using functional method
        - all err vars are [up,low]
    '''
    mix = mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq,lam,m_b)
    err1_up = abs(mixing(mt+mt_err[0],mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq,lam,m_b)-mix)
    err1_low = abs(mixing(mt+mt_err[1],mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq,lam,m_b)-mix)
    err2_up = abs(mixing(mt,mH,mW+mW_err[0],tanb,Vtq,Vtb,etaB,mB,fBq,BBq,lam,m_b)-mix)
    err2_low = abs(mixing(mt,mH,mW+mW_err[1],tanb,Vtq,Vtb,etaB,mB,fBq,BBq,lam,m_b)-mix)
    err3_up = abs(mixing(mt,mH,mW,tanb,Vtq+Vtq_err[0],Vtb,etaB,mB,fBq,BBq,lam,m_b)-mix)
    err3_low = abs(mixing(mt,mH,mW,tanb,Vtq+Vtq_err[1],Vtb,etaB,mB,fBq,BBq,lam,m_b)-mix)
    err4_up = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb+Vtb_err[0],etaB,mB,fBq,BBq,lam,m_b)-mix)
    err4_low = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb+Vtb_err[1],etaB,mB,fBq,BBq,lam,m_b)-mix)
    err5_up = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB+etaB_err[0],mB,fBq,BBq,lam,m_b)-mix)
    err5_low = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB+etaB_err[1],mB,fBq,BBq,lam,m_b)-mix)
    err6_up = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB+mB_err[0],fBq,BBq,lam,m_b)-mix)
    err6_low = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB+mB_err[1],fBq,BBq,lam,m_b)-mix)
    err7_up = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq+fBq_err[0],BBq,lam,m_b)-mix)
    err7_low = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq+fBq_err[1],BBq,lam,m_b)-mix)
    err8_up = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq+BBq_err[0],lam,m_b)-mix)
    err8_low = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq+BBq_err[1],lam,m_b)-mix)
    err10_up = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq,lam+lam_err[0],m_b)-mix)
    err10_low = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq,lam+lam_err[1],m_b)-mix)
    err11_up = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq,lam,m_b+m_b_err[0])-mix)
    err11_low = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq,lam,m_b+m_b_err[1])-mix)

    upper = np.sqrt(err1_up**2 + err2_up**2 + err3_up**2 + err4_up**2 + err5_up**2 + err6_up**2 + err7_up**2 + err8_up**2 + err10_up**2 + err11_up**2)
    lower = np.sqrt(err1_low**2 + err2_low**2 + err3_low**2 + err4_low**2 + err5_low**2 + err6_low**2 + err7_low**2 + err8_low**2 + err10_low**2 + err11_low**2)

    return upper, lower

########## KPI RATIOS ##########

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

########## R(D) ##########

def bsemi(mc,mb,m_B,m_D,p,d,mH,tanb):
    '''
        normalised branching ratio for B -> D tau nu, paramterisation used here for speed, but gives same result as full integration at 1.96 sigma (full things takes 1.5hrs to run so I have checked it but easier not to use here for convenience)
    '''
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

########## b to s gamma ##########

def bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb,Vcb,alp_EM,C):
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

    C1 = ((Vub/Vcb)**2)*(gamc/gamu) 
    R = ((Vts*Vtb/Vcb)**2)*(6*alp_EM/(np.pi*C1))*PplN # change C1 to C to use 0.546

    return R

def error_gamma(mt,mt_err,mW,mW_err,mub,lam_QCD,QCD_err,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamc_err,gamu,gamu_err,Vub,Vub_err,Vts,Vts_err,Vtb,Vtb_err,Vcb,Vcb_err,alp_EM,C,C_err):
    '''
        Calculates errors in branching ratios, using functional method
        - all err vars are [up,low]
    '''
    gams = bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb,Vcb,alp_EM,C)
    err1_up = abs(bsgamma(mt+mt_err[0],mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb,Vcb,alp_EM,C)-gams)
    err1_low = abs(bsgamma(mt+mt_err[1],mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb,Vcb,alp_EM,C)-gams)
    err2_up = abs(bsgamma(mt,mW+mW_err[0],mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb,Vcb,alp_EM,C)-gams)
    err2_low = abs(bsgamma(mt,mW+mW_err[1],mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb,Vcb,alp_EM,C)-gams)
    err3_up = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc+gamc_err[0],gamu,Vub,Vts,Vtb,Vcb,alp_EM,C)-gams)
    err3_low = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc+gamc_err[1],gamu,Vub,Vts,Vtb,Vcb,alp_EM,C)-gams)
    err4_up = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts+Vts_err[0],Vtb,Vcb,alp_EM,C)-gams)
    err4_low = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts+Vts_err[1],Vtb,Vcb,alp_EM,C)-gams)
    err5_up = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb+Vtb_err[0],Vcb,alp_EM,C)-gams)
    err5_low = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb+Vtb_err[1],Vcb,alp_EM,C)-gams)
    err6_up = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb,Vcb+Vcb_err[0],alp_EM,C)-gams)
    err6_low = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb,Vcb+Vcb_err[1],alp_EM,C)-gams)
    err7_up = abs(bsgamma(mt,mW,mub,lam_QCD+QCD_err[0],hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb,Vcb,alp_EM,C)-gams)
    err7_low = abs(bsgamma(mt,mW,mub,lam_QCD+QCD_err[1],hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb,Vcb,alp_EM,C)-gams)
    err8_up = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu+gamu_err[0],Vub,Vts,Vtb,Vcb,alp_EM,C)-gams)
    err8_low = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu+gamu_err[1],Vub,Vts,Vtb,Vcb,alp_EM,C)-gams)
    err9_up = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub+Vub_err[0],Vts,Vtb,Vcb,alp_EM,C)-gams)
    err9_low = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub+Vub_err[1],Vts,Vtb,Vcb,alp_EM,C)-gams)
    err10_up = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb,Vcb,alp_EM,C+C_err[0])-gams)
    err10_low = abs(bsgamma(mt,mW,mub,lam_QCD,hi,a,mH,tanb,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc,gamu,Vub,Vts,Vtb,Vcb,alp_EM,C+C_err[1])-gams)

    upper = np.sqrt(err1_up**2 + err2_up**2 + err3_up**2 + err4_up**2 + err5_up**2 + err6_up**2 + err7_up**2 + err8_up**2 + err9_up**2 + err10_up**2)
    lower = np.sqrt(err1_low**2 + err2_low**2 + err3_low**2 + err4_low**2 + err5_low**2 + err6_low**2 + err7_low**2 + err8_low**2 + err9_low**2 + err10_low**2)

    return upper, lower

########## Bs to mu mu ##########

def bmumu(mt,taubs,fbs,Vtb,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wangle,higgs,v,Vus,Vub,Vcs,Vcb,mH0,a,QCD):
    '''
        Branching ratio of b(s/d) to mu mu
        tau in ps, f in MeV, mt in GeV
    '''
    def I0(b):
        i = (1-3*b)/(-1+b) + 2*(b**2)*np.log(b)/((b-1)**2)
        return i
    def I1(b):
        i = -1/(b-1) + b*np.log(b)/((b-1)**2)
        return i
    def I2(b):
        i = (1-b)*I1(b)-1
        return i
    def I3(a,b):
        i = (7*a-b)*b/(a-b) + 2*(b**2)*np.log(b)*(2*a**2 -b**2 -6*a+3*b+2*a*b)/((b-1)*(a-b)**2) - 6*(a**2)*b*np.log(a)/(a-b)**2
        return i
    def I4(a,b):
        if a < 3600 or b < 3600:
            i = 0
        else:
            i = np.sqrt(b*a**3)*np.log(a)/((a-1)*(a-b)) - np.sqrt(a*b**3)*np.log(b)/((b-1)*(a-b))
        return i
    def I5(a,b):
        if a < 3600 and b < 3600:
            i = -1
        elif b < 3600 and a > 3600:
            i = -1 + a*np.log(a)/(a-1)
        elif b > 3600 and a < 3600:
            i = -1 + b*np.log(b)/(b-1)
        else:
            i = -1+(a**2)*np.log(a)/((a-1)*(a-b)) - (b**2)*np.log(b)/((b-1)*(a-b))
        return i
    def I6(b):
        i = b*(b-1)*I1(b)
        return i
    def I7(b):
        i = -b*I1(b)
        return i
    lmu = 2*np.log(mW/QCD)
    as_m = (12*np.pi/(23*lmu))*(1 - (348/529)*(np.log(lmu)/lmu))
    lmuM = 2*np.log(mW/mt)
    amu = as_m/np.pi
    lmut = 2*np.log(mt/QCD)
    as_mt = (12*np.pi/(23*lmut))*(1 - (348/529)*(np.log(lmut)/lmut))
    lmuMt = 2*np.log(mt/mt)
    amut = as_mt/np.pi
    factor1 = 1 - (4/3 + lmuM)*amu - (9.125 + 419*lmuM/72 + (2/9)*lmuM**2)*amu**2 - (0.3125*lmuM**3 + 4.5937*lmuM**2 + 25.3188*lmuM + 81.825)*amu**3
    factor2 = 1 - (4/3 + lmuMt)*amut - (9.125 + 419*lmuMt/72 + (2/9)*lmuMt**2)*amut**2 - (0.3125*lmuMt**3 + 4.5937*lmuMt**2 + 25.3188*lmuMt + 81.825)*amut**3
    mtmu = mt*factor1
    mtmut = 163.1#mt*factor2

    M = 750 # M 2HDM param, choose as you wish
    cob,g2,b = 1/tanb,0.65,np.arctan(tanb)
    z1,z2,y,yh,yH0 = (mu/mH)**2,(mc/mH)**2,(mW/mH)**2,(mH/higgs)**2,(mH/mH0)**2
    z3t,z3w = (mtmut/mH)**2,(mtmu/mH)**2
    el = np.sqrt(4*np.pi/137)
    #cba,sba = np.sin(2*b),-np.sin(2*b)
    cba,sba = np.cos(b-a),np.sin(b-a)
    Lp = (yh*cba**2 + yH0*sba**2)*(-2*tanb*mmu/v)
    Lm = -1*Lp
    lamh = -((higgs**2)*(3*np.cos(a+b)+np.cos(a-3*b)) + 4*np.sin(2*b)*np.sin(b-a)*mH**2 - 4*np.cos(a+b)*M**2)/(2*np.sin(2*b)*v**2)
    lamH0 = -((mH0**2)*(3*np.sin(a+b)+np.sin(a-3*b)) + 4*np.sin(2*b)*np.cos(b-a)*mH**2 - 4*np.sin(a+b)*M**2)/(2*np.sin(2*b)*v**2)
    #lamh = -((higgs**2)*(-3*np.sin(2*b)+np.sin(6*b)) - 4*np.sin(2*b)*np.sin(2*b)*mH**2 + 4*np.sin(2*b)*M**2)/(2*np.sin(2*b)*v**2)
    #lamH0 = -((mH0**2)*(3*np.sin(2*b)-np.sin(6*b)) + 4*np.sin(2*b)*np.sin(2*b)*mH**2 - 4*np.sin(2*b)*M**2)/(2*np.sin(2*b)*v**2)

    C10_1 = (1/(2*el**2))*(abs(cob*mtmu/v)**2)*(I1(z3t)-1)
    C10P_1 = -(abs(tanb/v)**2)*(mb*ms/(2*el**2))*(I1(z3t)-1)
    CS_1 = -(np.conj(tanb*ms/v)/((g2**4)*wangle*Vts*Vtb))*(-(y/2)*Lp*(4*I1(z3w)*(mtmut/mb)*(z3w-1)-2*np.log((mb/mH)**2)*(Vts*Vtb*(abs(cob*mtmut/v)**2))-I0(z3w)*Vts*Vtb*(abs(cob*mtmut/v)**2)+4*I5(z3w,z3w)*Vts*Vtb*(abs(cob*mtmut/v)**2))+2*I4(z3w,z3w)*Vts*Vtb*(abs(cob*mtmut/v)**2)*Lm*y-Vts*Vtb*np.conj(cob*mtmut/v)*((y*z3w)**0.5)*(-(tanb*mmu/v)-np.conj(tanb*mmu/v))*(2*(1-I1(z3w))*cba*g2*sba*(yh-yH0)+I1(z3w)*(y**0.5)*(cba*yh*lamh/mH - sba*yH0*lamH0/mH))) #CP
    CSP_1 = (1/((g2**4)*wangle*Vts*Vtb))*(y*Lm*(-2*I1(z3w)*(mtmut/mb)*(z3w-1)*((tanb*mtmut*(mb**2)/v**3)*Vts*Vtb - (tanb*(ms**2)*mtmut/v**3)*Vts*Vtb)+2*np.log((mb/mH)**2)*((cob*(mtmut**2)*mb/v**3)*Vts*Vtb+((tanb*(mb**2)*mtmut/v**3)*Vts*Vtb-(tanb*(ms**2)*mtmut/v**3)*Vts*Vtb)*mtmut/mb)-(tanb*mb/v)*(I7(z3w)*(abs(tanb*ms/v)**2)*Vts*Vtb+2*I5(z3w,z3w)*(abs(cob*mtmut/v)**2)*Vts*Vtb))+2*I4(z3w,z3w)*(cob*(mtmut**2)*mb/v**3)*Vts*Vtb*Lp*y+(mtmut*mb/v**2)*Vts*Vtb*((y*z3w)**0.5)*(-(tanb*mmu/v)+np.conj(-tanb*mmu/v))*(2*(1-I1(z3w))*cba*g2*sba*(yh-yH0)+I1(z3w)*(y**0.5)*(cba*yh*lamh/mH - sba*yH0*lamH0/mH))) #CP
    CS_2 = (np.conj(-tanb*ms/v)/(wangle*g2**2))*((z3w/4)*np.log((mb/mH)**2)*Lp+(1/8)*I3(y,z3w)*Lp+I2(z3w)*(-tanb*mmu/v)) #CP
    CSP_2 = (-tanb*mb/(v*wangle*g2**2))*((z3w/2)*np.log((mb/mH)**2)*Lm-(1/2)*I6(z3w)*Lm+I2(z3w)*(-tanb*mmu/v)) #CP
    C10_2 = -(pow(mW*mmu,2)/(wangle*Vts*Vtb*(mH**2)*pow(g2*v,4)))*((mu**2)*Vus*Vub*I1(z1)+(ms**2)*Vcs*Vcb*I1(z2)+(mb**2)*Vts*Vtb*I1(z3w))
    C10P_2 = -(pow(mmu*mW,2)*ms*mb*(tanb**4)/(wangle*Vtb*Vts*(mH**2)*pow(g2*v,4)))*(Vus*Vub*I1(z1)+Vcs*Vcb*I1(z2)+Vts*Vtb*I1(z3w))

    C10 = -4.103+C10_1+C10_2
    C10P = 0+C10P_1+C10P_2
    CS = 0+CS_1+CS_2 
    CSP = 0+CSP_1+CSP_2

    rm = mmu/mbs
    fr = np.sqrt(1-(4*rm**2))
    pref = ((g_gev**2)*(wangle**2)*pow(mW,4)/(32*np.pi**5))*pow(Vtb*Vts,2)*fr*mbs*(fbs**2)*pow(2*mmu,2)*taubs
    bs1 = pow(abs((mbs**2)*(np.conj(CS)-np.conj(CSP))/((mb+ms)*(2*mmu)) - (np.conj(C10)-np.conj(C10P))),2)
    bs2 = pow(abs((mbs**2)*(np.conj(CS)-np.conj(CSP))/((mb+ms)*(2*mmu))),2)*(1-(4*rm**2))
    bs = pref*(bs1+bs2)

    return bs

def error_bmumu(mt,mt_err,taubs,taubs_err,fbs,fbs_err,Vtd,Vtd_err,Vts,Vts_err,mmu,mmu_err,mbs,mbs_err,mW,mW_err,tanb,mH,mb,mb_err,ms,ms_err,mc,mc_err,mu,mu_err,wan,wan_err,higgs,higgs_err,v,v_err,Vus,Vus_err,Vub,Vub_err,Vcs,Vcs_err,Vcb,Vcb_err,MH0,a,QCD,QCD_err):
    '''
        Error propagation for b(s/d) to mumu
    '''
    bs = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)

    ## errors
    err1_up = bmumu(mt+mt_err[0],taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err1_low = bmumu(mt+mt_err[1],taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err2_up = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb+mb_err[0],ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err2_low = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb+mb_err[1],ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err3_up = bmumu(mt,taubs+taubs_err[0],fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err3_low = bmumu(mt,taubs+taubs_err[1],fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err4_up = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms+ms_err[0],mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err4_low = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms+ms_err[1],mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err5_up = bmumu(mt,taubs,fbs+fbs_err[0],Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err5_low = bmumu(mt,taubs,fbs+fbs_err[1],Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err6_up = bmumu(mt,taubs,fbs,Vtd+Vtd_err[0],Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err6_low = bmumu(mt,taubs,fbs,Vtd+Vtd_err[1],Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err7_up = bmumu(mt,taubs,fbs,Vtd,Vts+Vtd_err[0],mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err7_low = bmumu(mt,taubs,fbs,Vtd,Vts+Vts_err[1],mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err8_up = bmumu(mt,taubs,fbs,Vtd,Vts,mmu+mmu_err[0],mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err8_low = bmumu(mt,taubs,fbs,Vtd,Vts,mmu+mmu_err[1],mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err9_up = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc+mc_err[0],mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err9_low = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc+mc_err[1],mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err10_up = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs+mbs_err[0],mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err10_low = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs+mbs_err[1],mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err11_up = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW+mW_err[0],tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err11_low = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW+mW_err[1],tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err12_up = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu+mu_err[0],wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err12_low = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu+mu_err[1],wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err13_up = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan+wan_err[0],higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err13_low = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan+wan_err[1],higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err14_up = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs+higgs_err[0],v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err14_low = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs+higgs_err[1],v,Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err15_up = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v+v_err[0],Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err15_low = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v+v_err[1],Vus,Vub,Vcs,Vcb,MH0,a,QCD)
    err16_up = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus+Vus_err[0],Vub,Vcs,Vcb,MH0,a,QCD)
    err16_low = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus+Vus_err[1],Vub,Vcs,Vcb,MH0,a,QCD)
    err17_up = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub+Vub_err[0],Vcs,Vcb,MH0,a,QCD)
    err17_low = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub+Vub_err[1],Vcs,Vcb,MH0,a,QCD)
    err18_up = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs+Vcs_err[0],Vcb,MH0,a,QCD)
    err18_low = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs+Vcs_err[1],Vcb,MH0,a,QCD)
    err19_up = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb+Vcb_err[0],MH0,a,QCD)
    err19_low = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb+Vcb_err[1],MH0,a,QCD)
    err20_up = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD+QCD_err[0])
    err20_low = bmumu(mt,taubs,fbs,Vtd,Vts,mmu,mbs,mW,tanb,mH,mb,ms,mc,mu,wan,higgs,v,Vus,Vub,Vcs,Vcb,MH0,a,QCD+QCD_err[1])

    ## bs
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
    err17_up1, err17_low1 = abs(err17_up-bs),abs(err17_low-bs)
    err18_up1, err18_low1 = abs(err18_up-bs),abs(err18_low-bs)
    err19_up1, err19_low1 = abs(err19_up-bs),abs(err19_low-bs)
    err20_up1, err20_low1 = abs(err20_up-bs),abs(err20_low-bs)

    upper1 = np.sqrt(err1_up1**2 + err2_up1**2 + err3_up1**2 + err4_up1**2 + err5_up1**2 + err6_up1**2 + err7_up1**2 + err8_up1**2 + err9_up1**2 + err10_up1**2 + err11_up1**2 + err12_up1**2 + err13_up1**2 + err14_up1**2 + err15_up1**2 + err16_up1**2 + err17_up1**2 + err18_up1**2 + err19_up1**2 + err20_up1**2)
    lower1 = np.sqrt(err1_low1**2 + err2_low1**2 + err3_low1**2 + err4_low1**2 + err5_low1**2 + err6_low1**2 + err7_low1**2 + err8_low1**2 + err9_low1**2 + err10_low1**2 + err11_low1**2 + err12_low1**2 + err13_low1**2 + err14_low1**2 + err15_low1**2 + err16_low1**2 + err17_low1**2 + err18_low1**2 + err19_low1**2 + err20_low1**2)

    return upper1, lower1

    #oblique parameter functions 
def t(x,y,z):
    t=x+y-z
    return t
def r(x,y,z):
    r=z**2-2*z*(x+y)+(x-y)**2
    return r

def f(t,r):
    if r>0:
        f=(r**0.5)*np.log(np.absolute((t-r**0.5)/(t+r**0.5)))
    if r==0:
        f=0
    if r<0:
        f=2*(-r)**0.5*np.arctan((-r)**0.5/t)
    return f

def gxneqy(x,y,z):
    gxneqy=-16/3+5*(x+y)/z-(2*(x-y)**2)*z**(-2)+(3/z)*((x**2+y**2)/(x-y)-((x**2-y**2)/z)+(x-y)**3/(3*z**2))*np.log(x/y)+r(x,y,z)*f(t(x,y,z),r(x,y,z))/(z**3)
    return gxneqy

def g(x,y,z):
    if x!=y and x+y!=z:
        g=gxneqy(x,y,z)
    if x==y:
        g=gxneqy(x+1,x,z)  #small numerical aproximation to take limits
    elif x+y==z:
        g=gxneqy(x,y,x+y+1)
    elif y==z:
        g=gxneqy(x,z+1,z)
    return g

def gTilde(x,y,z):
    gTilde=-2+((x-y)/z-(x+y)/(x-y))*np.log(x/y)+f(t(x,y,z),r(x,y,z))/z
    return gTilde

def gTildeFixed(x,y,z):
    if x!=y:
        gTildeFixed=gTilde(x,y,z)
    if x==y:
        gTildeFixed=gTilde(x,x+0.1,z)
    return gTildeFixed

def gHat(x,z):
    gHat=g(x,z,z)+12*gTildeFixed(x,z,z)
    return gHat

def S2HDMofTheta (mHpm,mA0,mH0,Theta,mW,mZ,mh,Gf,alphaem,wangle):
    S2HDMofTheta=(wangle*Gf*mW**2/(alphaem*12*2**(0.5)*np.pi**2))\
    *(((2*wangle-1)**2)*g(mHpm**2,mHpm**2,mZ**2)\
    +(np.sin(Theta))**2*g(mA0**2,mH0**2,mZ**2)\
    +(np.cos(Theta))**2*g(mA0**2,mh**2,mZ**2)\
    +2*np.log(mA0*mH0*(mHpm**(-2)))\
    +(np.cos(Theta))**2*(gHat(mH0**2,mZ**2)-gHat(mh**2,mZ**2)))
    #S2HDMofTheta=(wangle*Gf*mW**2/(alphaem*12*2**(0.5)*np.pi**2))\
    #*(((2*wangle-1)**2)*g(mHpm**2,mHpm**2,mZ**2)\
    #+(-np.sin(2*Theta))**2*g(mA0**2,mH0**2,mZ**2)\
    #+(np.sin(2*Theta))**2*g(mA0**2,mh**2,mZ**2)\
    #+2*np.log(mA0*mH0*(mHpm**(-2)))\
    #+(np.sin(2*Theta))**2*(gHat(mH0**2,mZ**2)-gHat(mh**2,mZ**2)))
    return S2HDMofTheta

def S2HDMofAlphaBeta (mHpm,mA0,mH0,Alpha,Beta,mW,mZ,mh,Gf,alphaem,wangle):
    S2HDMofAlphaBeta=S2HDMofTheta(mHpm,mA0,mH0,Beta-Alpha,mW,mZ,mh,Gf,alphaem,wangle)
    #S2HDMofAlphaBeta=S2HDMofTheta(mHpm,mA0,mH0,Beta,mW,mZ,mh,Gf,alphaem,wangle)
    return S2HDMofAlphaBeta

def SOb_err(mHp,mA0,mH0,alpha,beta,mW,mW_err,mZ,mZ_err,mh,mh_err,Gf,alphaem,wangle,wan_err):
    S = S2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ,mh,Gf,alphaem,wangle)

    err1_up = abs(S2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW+mW_err[0],mZ,mh,Gf,alphaem,wangle)-S)
    err1_lo = abs(S2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW+mW_err[1],mZ,mh,Gf,alphaem,wangle)-S)
    err2_up = abs(S2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ+mZ_err[0],mh,Gf,alphaem,wangle)-S)
    err2_lo = abs(S2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ+mZ_err[1],mh,Gf,alphaem,wangle)-S)
    err3_up = abs(S2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ,mh+mh_err[0],Gf,alphaem,wangle)-S)
    err3_lo = abs(S2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ,mh+mh_err[1],Gf,alphaem,wangle)-S)
    err4_up = abs(S2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ,mh,Gf,alphaem,wangle+wan_err[0])-S)
    err4_lo = abs(S2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ,mh,Gf,alphaem,wangle+wan_err[1])-S)

    upper = np.sqrt(err1_up**2 + err2_up**2 + err3_up**2 + err4_up**4)
    lower = np.sqrt(err1_lo**2 + err2_lo**2 + err3_lo**2 + err4_lo**4)

    return upper, lower

def U2HDMofTheta (mHpm,mA0,mH0,Theta,mW,mZ,mh,Gf,alphaem,wangle):
    U2HDMofTheta=((Gf*mW**2)/(48*2**0.5*np.pi**2*alphaem))*(g(mHpm**2,mA0**2,mW**2)\
    +(np.sin(Theta))**2*g(mHpm**2,mH0**2,mW**2)+(np.cos(Theta))**2*g(mHpm**2,mh**2,mW**2)\
    -(2*wangle-1)**2*g(mHpm**2,mHpm**2,mZ**2)\
    -(np.sin(Theta))**2*g(mA0**2,mH0**2,mZ**2)\
    -(np.cos(Theta))**2*g(mA0**2,mh**2,mZ**2)\
    +(np.cos(Theta))**2*(gHat(mH0**2,mW**2)-gHat(mH0**2,mZ**2))\
    -(np.cos(Theta))**2*(gHat(mh**2,mW**2)-gHat(mh**2,mZ**2)))
    #U2HDMofTheta=((Gf*mW**2)/(48*2**0.5*np.pi**2*alphaem))*(g(mHpm**2,mA0**2,mW**2)\
    #+(-np.sin(2*Theta))**2*g(mHpm**2,mH0**2,mW**2)+(np.sin(2*Theta))**2*g(mHpm**2,mh**2,mW**2)\
    #-(2*wangle-1)**2*g(mHpm**2,mHpm**2,mZ**2)\
    #-(-np.sin(2*Theta))**2*g(mA0**2,mH0**2,mZ**2)\
    #-(np.sin(2*Theta))**2*g(mA0**2,mh**2,mZ**2)\
    #+(np.sin(2*Theta))**2*(gHat(mH0**2,mW**2)-gHat(mH0**2,mZ**2))\
    #-(np.sin(2*Theta))**2*(gHat(mh**2,mW**2)-gHat(mh**2,mZ**2)))
    return U2HDMofTheta

def U2HDMofAlphaBeta (mHpm,mA0,mH0,Alpha,Beta,mW,mZ,mh,Gf,alphaem,wangle):
    U2HDMofAlphaBetta=U2HDMofTheta(mHpm,mA0,mH0,Beta-Alpha,mW,mZ,mh,Gf,alphaem,wangle)
    #U2HDMofAlphaBetta=U2HDMofTheta(mHpm,mA0,mH0,Beta,mW,mZ,mh,Gf,alphaem,wangle)
    return U2HDMofAlphaBetta

def UOb_err(mHp,mA0,mH0,alpha,beta,mW,mW_err,mZ,mZ_err,mh,mh_err,Gf,alphaem,wangle,wan_err):
    U = U2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ,mh,Gf,alphaem,wangle)

    err1_up = abs(U2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW+mW_err[0],mZ,mh,Gf,alphaem,wangle)-U)
    err1_lo = abs(U2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW+mW_err[1],mZ,mh,Gf,alphaem,wangle)-U)
    err2_up = abs(U2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ+mZ_err[0],mh,Gf,alphaem,wangle)-U)
    err2_lo = abs(U2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ+mZ_err[1],mh,Gf,alphaem,wangle)-U)
    err3_up = abs(U2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ,mh+mh_err[0],Gf,alphaem,wangle)-U)
    err3_lo = abs(U2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ,mh+mh_err[1],Gf,alphaem,wangle)-U)
    err4_up = abs(U2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ,mh,Gf,alphaem,wangle+wan_err[0])-U)
    err4_lo = abs(U2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ,mh,Gf,alphaem,wangle+wan_err[1])-U)

    upper = np.sqrt(err1_up**2 + err2_up**2 + err3_up**2 + err4_up**4)
    lower = np.sqrt(err1_lo**2 + err2_lo**2 + err3_lo**2 + err4_lo**4)

    return upper, lower

def F(x,y):
    if x!=y:
        F=(x+y)/2-((x*y)*(x-y)**(-1))*np.log(x/y)
    elif x==y:
        F=0
    return F

def T(MHpm,MA0,MH0,Theta,mW,mZ,mh,Gf,alphaem):
    #T=(Gf/(8*(2**0.5)*alphaem*(np.pi)**2))*(F(MHpm**2,MA0**2)+(-np.sin(2*Theta))**2*F(MHpm**2,MH0**2)\
    #+((np.sin(2*Theta))**2)*F(MHpm**2,mh**2)-((-np.sin(2*Theta))**2)*F(MA0**2,MH0**2)\
    #-((np.sin(2*Theta))**2)*F(MA0**2,mh**2)\
    #+3*((np.sin(2*Theta))**2)*(F(mZ**2,MH0**2)-F(mW**2,MH0**2))\
    #-3*((np.sin(2*Theta))**2)*(F(mZ**2,mh**2)-F(mW**2,mh**2)))
    T=(Gf/(8*(2**0.5)*alphaem*(np.pi)**2))*(F(MHpm**2,MA0**2)+(np.sin(Theta))**2*F(MHpm**2,MH0**2)\
    +((np.cos(Theta))**2)*F(MHpm**2,mh**2)-((np.sin(Theta))**2)*F(MA0**2,MH0**2)\
    -((np.cos(Theta))**2)*F(MA0**2,mh**2)\
    +3*((np.cos(Theta))**2)*(F(mZ**2,MH0**2)-F(mW**2,MH0**2))\
    -3*((np.cos(Theta))**2)*(F(mZ**2,mh**2)-F(mW**2,mh**2)))
    return T

def T2HDMofAlphaBeta(MHpm,MA0,MH0,Alpha,Beta,mW,mZ,mh,Gf,alphaem):
    Tofalphabeta = T(MHpm,MA0,MH0,Beta-Alpha,mW,mZ,mh,Gf,alphaem)
    #Tofalphabeta = T(MHpm,MA0,MH0,Beta,mW,mZ,mh,Gf,alphaem)
    return Tofalphabeta

def TOb_err(mHp,mA0,mH0,alpha,beta,mW,mW_err,mZ,mZ_err,mh,mh_err,Gf,alphaem):
    T = T2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ,mh,Gf,alphaem)

    err1_up = abs(T2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW+mW_err[0],mZ,mh,Gf,alphaem)-T)
    err1_lo = abs(T2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW+mW_err[1],mZ,mh,Gf,alphaem)-T)
    err2_up = abs(T2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ+mZ_err[0],mh,Gf,alphaem)-T)
    err2_lo = abs(T2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ+mZ_err[1],mh,Gf,alphaem)-T)
    err3_up = abs(T2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ,mh+mh_err[0],Gf,alphaem)-T)
    err3_lo = abs(T2HDMofAlphaBeta(mHp,mA0,mH0,alpha,beta,mW,mZ,mh+mh_err[1],Gf,alphaem)-T)

    upper = np.sqrt(err1_up**2 + err2_up**2 + err3_up**2)
    lower = np.sqrt(err1_lo**2 + err2_lo**2 + err3_lo**2)

    return upper, lower

########## GLOBAL ##########

def itera_global(
        bpls_exp,bpls_exp_error,dpls_exp,dpls_exp_error,dspls_exp,dspls_exp_error,
        bpmu,bpmu_err,dsmu,dsmu_err,bmix_exp,bmix_exp_error,bmixs_exp,bmixs_exp_error,
        kpi_exp,kpi_exp_error,tkpi_exp,tkpi_exp_error,
        bsmu_exp,bsmu_exp_err,bdmu_exp,bdmu_exp_err,rd_exp,rd_exp_err,rdst_exp,rdst_exp_err,
        gams_exp,gams_exp_error,gamc_exp,gamc_exp_error,gamu,gamu_err,
        mu,mu_err,md,md_err,mc,mc_err,ms,ms_err,mb,mb_err,mt,mt_err,mW,mW_err,
        mbpls,mbpls_err,mdpls,mdpls_err,mdspls,mdspls_err,mbd,mbd_err,mbs,mbs_err,mdst,mdst_err,
        mK,mK_err,mpi,mpi_err,mtau,mtau_err,mmu,mmu_err,
        Vud,Vud_err,Vus,Vus_err,Vub,Vub_err,Vcd,Vcd_err,Vcs,Vcs_err,Vcb,Vcb_err,Vtd,Vtd_err,Vts,Vts_err,Vtb,Vtb_err,
        etaB,etaB_err,f2Bd,f2Bd_err,f2Bs,f2Bs_err,fBs,fBs_err,Bbd,Bbd_err,BBs,BBs_err,
        fbpls,fbpls_err,fdpls,fdpls_err,fdspls,fdspls_err,
        fKpi,fKpi_err,delt_kpi,delt_kpi_err,delt_tau,delt_tau_err,
        tbpls,tbpls_err,tdpls,tdpls_err,tdspls,tdspls_err,tbd,tbd_err,tbs,tbs_err,
        mub,hi,a,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,alp_EM,C,C_err,
        p,p_err,ps,ps_err,d,d_err,r01,r01_err,r11,r11_err,r21,r21_err,
        delta_b,delta_d,
        wangle,wangle_err,lam_QCD,QCD_err,higgs,higgs_err,vev,vev_err,
        SOblique,SOblique_err,TOblique,TOblique_err,UOblique,UOblique_err,mZ,mZ_err):
    '''
    Iterate of mH,tanb space for everything

    Inputs: 
        Shape and types of inputs can be seen in global.py - mostly single numbers for inputs, then lists of +/- error for errors.

        - Exp values
            bpls_exp,bpls_exp_error,dpls_exp,dpls_exp_error,dspls_exp,dspls_exp_error,
            bpmu,bpmu_err,dsmu,dsmu_err,bmix_exp,bmix_exp_error,bmixs_exp,bmixs_exp_error,
            kpi_exp,kpi_exp_error,tkpi_exp,tkpi_exp_error,
            bsmu_exp,bsmu_exp_err,bdmu_exp,bdmu_exp_err,rd_exp,rd_exp_err,
            gams_exp,gams_exp_error,gamc_exp,gamc_exp_error,gamu,gamu_err,
        
        - Masses
            mu,mu_err,md,md_err,mc,mc_err,ms,ms_err,mb,mb_err,mt,mt_err,mW,mW_err,
            mbpls,mbpls_err,mdpls,mdpls_err,mdspls,mdspls_err,mbd,mbd_err,mbs,mbs_err,
            mK,mK_err,mpi,mpi_err,mtau,mtau_err,mmu,mmu_err,
        
        - CKM Elements (Mod)
            Vud,Vud_err,Vus,Vus_err,Vub,Vub_err,Vcd,Vcd_err,Vcs,Vcs_err,
            Vcb,Vcb_err,Vtd,Vtd_err,Vts,Vts_err,Vtb,Vtb_err,
        
        - Decay constants and Bag parameters
            etaB,etaB_err,f2Bd,f2Bd_err,f2Bs,f2Bs_err,fBs,fBs_err,Bbd,Bbd_err,BBs,BBs_err,
            fbpls,fbpls_err,fdpls,fdpls_err,fdspls,fdspls_err,
            fKpi,fKpi_err,delt_kpi,delt_kpi_err,delt_tau,delt_tau_err,
        
        - Lifetimes
            tbpls,tbpls_err,tdpls,tdpls_err,tdspls,tdspls_err,tbd,tbd_err,tbs,tbs_err,

        - bsgamma values
            mub,hi,a,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,alp_EM,C,C_err

        - R(D) params, radiative leptonic corrections and misc
            p,p_err,d,d_err, 
            delta_b,delta_d,
            wangle,wangle_err,lam_QCD,QCD_err,higgs,higgs_err,vev,vev_err

        - Oblique params
            SOblique,SOblique_err,TOblique,TOblique_err,UOblique,UOblique_err,mZ,mZ_err

    Outputs:
        - 3 lists of arrays, in order: (semi-)leptonics; b mixing; bsgamma; combine first three; Bs to mumu; R(D); everything
        1) [mH...], a list of allowed mH value arrays 
        2) [tanb...], a list of allowed tanb value arrays
        3) [chis...], a list of chisq value arrays for each mH,tanb point
        - list of [minimum chisq value,mH coord,tanb coord] for complete fit

    Adding a new observable:
        
        If function for observable is in separate file, import file at the start of global.py, or copy it into this file

        Add to Inputs: observ_exp, observ_exp_error, any other new parameters
        
        - Add error limits for experiment, balanced to be Gaussian and multiplied by sigma
            observ_exp_up,observ_exp_down= observ_exp+observ_exp_error[0],observ_exp+observ_exp_error[1]
            av_observ = 0.5*(observ_exp_up+observ_exp_down)
            sige_observ = sigma*(observ_exp_up-av_observ)

        - Add lists to store mH, tanb, and chisq values 
          (if you want to output this by itself as well)pi
            mH_observ_loc, tanb_observ_loc, chisq_observs = [],[],[]

        - In for loop, calculate value, balance errors, set up boolean 
        (An example from how I am doing this, you may be doing e.g. errors differently, so just do that)
            observ_the = observ_func(parameters...,i,j,parameters...) (i = mH, j = tanb)
            observ_err = error_observ_func(parameters...,errors...,i,j,parameters...,j...)
            observ_the_up, observ_the_down = observ_the+observ_err[0],observ_the-observ_err[1]
            mid_observ = 0.5*(observ_the_up+observ_the_down)
            sig_observ = sigma*(observ_the_up-mid_observ),sigma*(observ_the_up-mid_observ)
            observ_bool = ((av_observ >= mid_observ and mid_observ+sig_observ >= av_observ-sige_observ) or (av_observ <= mid_observ and mid_observ-sig_observ <= av_observ+sige_observ))

        - Finding observable allowed region (if you want to output this by itself)
            if observ_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mH_observ_loc = np.append(mH_observ_loc,i_log)
                tanb_observ_loc = np.append(tanb_observ_loc,j_log)
                chi_obs_ij = chisq_simp([av_observ],[mid_observ],[sige_observ],[sig_observ])
                chi_observs = np.append(chi_observs,chi_obs_ij)

        - If outputting by itself, add mH_observ_loc, tanb_observ_loc, chisq_observs to output lists

        - Add to global if statement and append chisq_simp input lists
            if ... and observ_bool:
                i_log...
                mH...
                tanb...
                chi_2ij = chisq_simp([av_b,...,av_observ],[mid_b,...,mid_observ],[sige_b,...,sige_observ],[sig_b,...,sig_observ])

    Any issues, let me know. 
    '''
    sigma = 3 # to 95% CL

    ##### LEPTONICS #####
    bpls_exp_up,bpls_exp_down = bpls_exp+bpls_exp_error[0],bpls_exp+bpls_exp_error[1]
    bpmu_exp_up,bpmu_exp_down = bpmu+bpmu_err[0],bpmu+bpmu_err[1]
    dpls_exp_up,dpls_exp_down = dpls_exp+dpls_exp_error[0],dpls_exp+dpls_exp_error[1]
    dspls_exp_up,dspls_exp_down = dspls_exp+dspls_exp_error[0],dspls_exp+dspls_exp_error[1]
    dsmu_exp_up,dsmu_exp_down = dsmu+dsmu_err[0],dsmu+dsmu_err[1]
    av_b,av_d,av_ds = 0.5*(bpls_exp_up+bpls_exp_down),0.5*(dpls_exp_up+dpls_exp_down),0.5*(dspls_exp_up+dspls_exp_down)
    av_bm,av_dm = 0.5*(bpmu_exp_up+bpmu_exp_down),0.5*(dsmu_exp_up+dsmu_exp_down)
    sige_b,sige_d,sige_ds = sigma*(bpls_exp_up-av_b),sigma*(dpls_exp_up-av_d),sigma*(dspls_exp_up-av_ds)
    sige_bm,sige_dm = sigma*(bpmu_exp_up-av_bm),sigma*(dsmu_exp_up-av_dm)

    ##### MIXING #####
    bmix_exp_up,bmix_exp_down = bmix_exp+bmix_exp_error[0],bmix_exp+bmix_exp_error[1]
    bmixs_exp_up,bmixs_exp_down = bmixs_exp+bmixs_exp_error[0],bmixs_exp+bmixs_exp_error[1]
    av_bmix,av_bmixs = 0.5*(bmix_exp_up+bmix_exp_down),0.5*(bmixs_exp_up+bmixs_exp_down)
    sige_bmix,sige_bmixs = sigma*(bmix_exp_up-av_bmix),sigma*(bmixs_exp_up-av_bmixs)

    ##### K/PI RATIOS #####
    kpi_exp_up,kpi_exp_down = kpi_exp+kpi_exp_error[0],kpi_exp+kpi_exp_error[1]
    tkpi_exp_up,tkpi_exp_down = tkpi_exp+tkpi_exp_error[0],tkpi_exp+tkpi_exp_error[1]
    av_k,av_t = 0.5*(kpi_exp_up+kpi_exp_down),0.5*(tkpi_exp_up+tkpi_exp_down)
    sige_k,sige_t = sigma*(kpi_exp_up-av_k),sigma*(tkpi_exp_up-av_t)

    ##### BSGAMMA #####
    gam_exp = gams_exp/gamc_exp
    xgam = gam_exp*np.sqrt((gamc_exp_error[0]/gamc_exp)**2 + (gams_exp_error[0]/gams_exp)**2)
    ygam = gam_exp*np.sqrt((gamc_exp_error[1]/gamc_exp)**2 + (gams_exp_error[1]/gams_exp)**2)
    gam_exp_up,gam_exp_down = gam_exp+xgam,gam_exp-ygam
    av_g = 0.5*(gam_exp_up+gam_exp_down)
    sige_g = sigma*(gam_exp_up-av_g)

    ##### B TO MU MU #####
    bsmu_exp_up,bsmu_exp_down = bsmu_exp+bsmu_exp_err[0],bsmu_exp+bsmu_exp_err[1]
    av_bsmu = 0.5*(bsmu_exp_up+bsmu_exp_down)
    sige_bsmu = sigma*(bsmu_exp_up-av_bsmu)

    bdmu_exp_up,bdmu_exp_down = bdmu_exp+bdmu_exp_err[0],bdmu_exp+bdmu_exp_err[1]
    av_bdmu = 0.5*(bdmu_exp_up+bdmu_exp_down)
    sige_bdmu = sigma*(bdmu_exp_up-av_bdmu)

    ##### R(D) #####
    rd_exp_up,rd_exp_down = rd_exp+rd_exp_err[0],rd_exp+rd_exp_err[1]
    rds_exp_up,rds_exp_down = rdst_exp+rdst_exp_err[0],rdst_exp+rdst_exp_err[1]
    av_rd,av_rds = 0.5*(rd_exp_up+rd_exp_down),0.5*(rds_exp_up+rds_exp_down)
    sige_rd,sige_rds = sigma*(rd_exp_up-av_rd),sigma*(rds_exp_up-av_rds)

    #Oblique
    SOblique_up,SOblique_down=SOblique+SOblique_err[0],SOblique+SOblique_err[1]
    av_SOblique=0.5*(SOblique_up+SOblique_down)
    sige_SOblique=sigma*(SOblique_up-av_SOblique)

    TOblique_up,TOblique_down=TOblique+TOblique_err[0],TOblique+TOblique_err[1]
    av_TOblique=0.5*(TOblique_up+TOblique_down)
    sige_TOblique=sigma*(TOblique_up-av_TOblique)

    UOblique_up,UOblique_down=UOblique+UOblique_err[0],UOblique+UOblique_err[1]
    av_UOblique=0.5*(UOblique_up+UOblique_down)
    sige_UOblique=sigma*(UOblique_up-av_UOblique)

    ##### CHI SQUARED LISTS #####
    chi_ls,chi_ms,chi_gs,chi_mus,chi_1s,chi_2s,chi_Us,chi_Ts,chi_Ss=[],[],[],[],[],[],[],[],[]
    chi_2min = [100,0,0] #so we can get location in mH, tanb space for minimum chisq for global fit

    log_mH_range = np.linspace(1,3.5,250)
    log_tanb_range = np.linspace(-1,2,300)
    mH_range = 10**log_mH_range
    tanb_range = 10**log_tanb_range
    mHl_loc,mHb_loc,mHg_loc,mHa_loc,mHmu_loc,mHa2_loc,mHS_loc,mHT_loc,mHU_loc = [],[],[],[],[],[],[],[],[]
    tanbl_loc,tanbb_loc,tanbg_loc,tanba_loc,tanbmu_loc,tanba2_loc,tanbS_loc,tanbT_loc,tanbU_loc = [],[],[],[],[],[],[],[],[]
    for i in mH_range:
        for j in tanb_range:
            b = np.arctan(j)
            alph = b - np.pi/2 # find alpha in alignment limit
#            alph = b - np.arccos(0.465) # find alpha in wrong sign limit
            mH0 = 1500 # set H0 and A0 masses
            mA0 = 1500
            ##### LEPTONICS #####
            bpls_the, dpls_the, dspls_the = bthe(mbpls,mtau,Vub,fbpls,tbpls,mu,mb,j,i,1),bthe(mdpls,mmu,Vcd,fdpls,tdpls,mc,md,j,i,delta_d),bthe(mdspls,mtau,Vcs,fdspls,tdspls,mc,ms,j,i,1)
            bpmu_the, dsmu_the = bthe(mbpls,mmu,Vub,fbpls,tbpls,mu,mb,j,i,1),bthe(mdspls,mmu,Vcs,fdspls,tdspls,mc,ms,j,i,1)
            bpls_err, dpls_err, dspls_err = error_branching(mbpls,mbpls_err,mtau,mtau_err,Vub,Vub_err,fbpls,fbpls_err,tbpls,tbpls_err,mu,mu_err,mb,mb_err,j,i,1),error_branching(mdpls,mdpls_err,mmu,mmu_err,Vcd,Vcd_err,fdpls,fdpls_err,tdpls,tdpls_err,mc,mc_err,md,md_err,j,i,delta_d),error_branching(mdspls,mdspls_err,mtau,mtau_err,Vcs,Vcs_err,fdspls,fdspls_err,tdspls,tdspls_err,mc,mc_err,ms,ms_err,j,i,1)
            bpmu_err, dsmu_err = error_branching(mbpls,mbpls_err,mmu,mmu_err,Vub,Vub_err,fbpls,fbpls_err,tbpls,tbpls_err,mu,mu_err,mb,mb_err,j,i,1),error_branching(mdspls,mdspls_err,mmu,mmu_err,Vcs,Vcs_err,fdspls,fdspls_err,tdspls,tdspls_err,mc,mc_err,ms,ms_err,j,i,1)
            bpls_the_up,bpls_the_down,dpls_the_up,dpls_the_down,dspls_the_up,dspls_the_down=bpls_the+bpls_err[0],bpls_the-bpls_err[1],dpls_the+dpls_err[0],dpls_the-dpls_err[1],dspls_the+dspls_err[0],dspls_the-dspls_err[1]
            bpmu_the_up,bpmu_the_down,dsmu_the_up,dsmu_the_down=bpmu_the+bpmu_err[0],bpmu_the-bpmu_err[1],dsmu_the+dsmu_err[0],dsmu_the-dsmu_err[1]
            mid_b,mid_d,mid_ds=0.5*(bpls_the_up+bpls_the_down),0.5*(dpls_the_up+dpls_the_down),0.5*(dspls_the_up+dspls_the_down)
            mid_bmu,mid_dm=0.5*(bpmu_the_up+bpmu_the_down),0.5*(dsmu_the_up+dsmu_the_down)
            sig_b,sig_d,sig_ds=sigma*(bpls_the_up-mid_b),sigma*(dpls_the_up-mid_d),sigma*(dspls_the_up-mid_ds)
            sig_bmu,sig_dm=sigma*(bpmu_the_up-mid_bmu),sigma*(dsmu_the_up-mid_dm)
            bpls_bool = ((av_b >= mid_b and mid_b+sig_b >= av_b-sige_b) or (av_b <= mid_b and mid_b-sig_b <= av_b+sige_b)) 
            bpmu_bool = ((av_bm >= mid_bmu and mid_bmu+sig_bmu >= av_bm-sige_bm) or (av_bm <= mid_bmu and mid_bmu-sig_bmu <= av_bm+sige_bm)) 
            dpls_bool = ((av_d >= mid_d and mid_d+sig_d >= av_d-sige_d) or (av_d <= mid_d and mid_d-sig_d <= av_d+sige_d))
            dspls_bool = ((av_ds >= mid_ds and mid_ds+sig_ds >= av_ds-sige_ds) or (av_ds <= mid_ds and mid_ds-sig_ds <= av_ds+sige_ds))
            dsmu_bool = ((av_dm >= mid_dm and mid_dm+sig_dm >= av_dm-sige_dm) or (av_dm <= mid_dm and mid_dm-sig_dm <= av_dm+sige_dm))

            ##### MIXING #####
            # note - currently putting in fB^2*Bag as fB, change as needed
            bmix_the,bmixs_the = mixing(mt,i,mW,j,Vtd,Vtb,etaB,mbd,f2Bd,Bbd,lam_QCD,mb),mixing(mt,i,mW,j,Vts,Vtb,etaB,mbs,f2Bs,BBs,lam_QCD,mb)
            bmix_err,bmixs_err = error_mixing(mt,mt_err,i,mW,mW_err,j,Vtd,Vtd_err,Vtb,Vtb_err,etaB,etaB_err,mbd,mbd_err,f2Bd,f2Bd_err,Bbd,Bbd_err,lam_QCD,QCD_err,mb,mb_err),error_mixing(mt,mt_err,i,mW,mW_err,j,Vts,Vts_err,Vtb,Vtb_err,etaB,etaB_err,mbs,mbs_err,f2Bs,f2Bs_err,BBs,BBs_err,lam_QCD,QCD_err,mb,mb_err)
            bmix_the_up, bmix_the_down = bmix_the+bmix_err[0],bmix_the-bmix_err[1]
            bmixs_the_up, bmixs_the_down = bmixs_the+bmixs_err[0],bmixs_the-bmixs_err[1]
            mid_bm,mid_bms=0.5*(bmix_the_up+bmix_the_down),0.5*(bmixs_the_up+bmixs_the_down)
            sig_bm,sig_bms=sigma*(bmix_the_up-mid_bm),sigma*(bmixs_the_up-mid_bms)
            bmix_bool = ((av_bmix >= mid_bm and mid_bm+sig_bm >= av_bmix-sige_bmix) or (av_bmix <= mid_bm and mid_bm-sig_bm <= av_bmix+sige_bmix)) and ((av_bmixs >= mid_bms and mid_bms+sig_bms >= av_bmixs-sige_bmixs) or (av_bmixs <= mid_bms and mid_bms-sig_bms <= av_bmixs+sige_bmixs))

            ##### K/PI RATIOS #####
            kpi_the,tkpi_the = decay_bsm(mK,mpi,mmu,mtau,Vus,Vud,fKpi,delt_kpi,delt_tau,ms,md,mu,j,i)
            kpi_uperr,kpi_derr,tkpi_uperr,tkpi_derr = error_kpi(mK,mK_err,mpi,mpi_err,mmu,mmu_err,mtau,mtau_err,Vus,Vus_err,Vud,Vud_err,fKpi,fKpi_err,delt_kpi,delt_kpi_err,delt_tau,delt_tau_err,ms,ms_err,md,md_err,mu,mu_err,j,i)
            kpi_the_up,kpi_the_down,tkpi_the_up,tkpi_the_down = kpi_the+kpi_uperr,kpi_the-kpi_derr,tkpi_the+tkpi_uperr,tkpi_the-tkpi_derr
            mid_k,mid_t=0.5*(kpi_the_up+kpi_the_down),0.5*(tkpi_the_up+tkpi_the_down)
            sig_k,sig_t=sigma*(kpi_the_up-mid_k),sigma*(tkpi_the_up-mid_t)
            kpi_bool = ((av_k >= mid_k and mid_k+sig_k >= av_k-sige_k) or (av_k <= mid_k and mid_k-sig_k <= av_k+sige_k)) 
            tkpi_bool = ((av_t >= mid_t and mid_t+sig_t >= av_t-sige_t) or (av_t <= mid_t and mid_t-sig_t <= av_t+sige_t)) 

            ##### BSGAMMA #####
            gam_the = bsgamma(mt,mW,mub,lam_QCD,hi,a,i,j,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc_exp,gamu,Vub,Vts,Vtb,Vcb,alp_EM,C)
            gam_err = error_gamma(mt,mt_err,mW,mW_err,mub,lam_QCD,QCD_err,hi,a,i,j,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc_exp,gamc_exp_error,gamu,gamu_err,Vub,Vub_err,Vts,Vts_err,Vtb,Vtb_err,Vcb,Vcb_err,alp_EM,C,C_err)
            gam_the_up,gam_the_down = gam_the+gam_err[0],gam_the-gam_err[1]
            mid_g=0.5*(gam_the_up+gam_the_down)
            sig_g=sigma*(gam_the_up-mid_g)
            gam_bool = ((av_g >= mid_g and mid_g+sig_g >= av_g-sige_g) or (av_g <= mid_g and mid_g-sig_g <= av_g+sige_g)) 

            ##### B TO MU MU #####
            expect_bs = bmumu(mt,tbs,fBs,Vtb,Vts,mmu,mbs,mW,j,i,mb,ms,mc,mu,wangle,higgs,vev,Vus,Vub,Vcs,Vcb,mH0,alph,lam_QCD)
            expect_bs_uperr,expect_bs_downerr = error_bmumu(mt,mt_err,tbs,tbs_err,fBs,fBs_err,Vtb,Vtb_err,Vts,Vts_err,mmu,mmu_err,mbs,mbs_err,mW,mW_err,j,i,mb,mb_err,ms,ms_err,mc,mc_err,mu,mu_err,wangle,wangle_err,higgs,higgs_err,vev,vev_err,Vus,Vus_err,Vub,Vub_err,Vcs,Vcs_err,Vcb,Vcb_err,mH0,alph,lam_QCD,QCD_err)
            expect_bs_up, expect_bs_down = expect_bs+expect_bs_uperr, expect_bs-expect_bs_downerr
            mid_bsmu=0.5*(expect_bs_up+expect_bs_down)
            sig_bsmu=sigma*(expect_bs_up-mid_bsmu)
            bs_bool = ((av_bsmu >= mid_bsmu and mid_bsmu+sig_bsmu >= av_bsmu-sige_bsmu) or (av_bsmu <= mid_bsmu and mid_bsmu-sig_bsmu <= av_bsmu+sige_bsmu)) 

            expect_bd = bmumu(mt,tbd,fbpls,Vtb,Vtd,mmu,mbd,mW,j,i,mb,ms,mc,mu,wangle,higgs,vev,Vus,Vub,Vcs,Vcb,mH0,alph,lam_QCD)
            expect_bd_uperr,expect_bd_downerr = error_bmumu(mt,mt_err,tbd,tbd_err,fbpls,fbpls_err,Vtb,Vtb_err,Vtd,Vtd_err,mmu,mmu_err,mbd,mbd_err,mW,mW_err,j,i,mb,mb_err,ms,ms_err,mc,mc_err,mu,mu_err,wangle,wangle_err,higgs,higgs_err,vev,vev_err,Vus,Vus_err,Vub,Vub_err,Vcs,Vcs_err,Vcb,Vcb_err,mH0,alph,lam_QCD,QCD_err)
            expect_bd_up, expect_bd_down = expect_bd+expect_bd_uperr, expect_bd-expect_bd_downerr
            mid_bdmu=0.5*(expect_bd_up+expect_bd_down)
            sig_bdmu=sigma*(expect_bd_up-mid_bdmu)
            bd_bool = ((av_bdmu >= mid_bdmu and mid_bdmu+sig_bdmu >= av_bdmu-sige_bdmu) or (av_bdmu <= mid_bdmu and mid_bdmu-sig_bdmu <= av_bdmu+sige_bdmu)) 

            ##### R(D) #####
            expect_rds_up,expect_rds_down = error_rds(mbd,mbd_err,mdst,mdst_err,ps,ps_err,r01,r01_err,r11,r11_err,r21,r21_err,Vcb,Vcb_err,mmu,mmu_err,mtau,mtau_err,vev,vev_err,mc,mc_err,mb,mb_err,j,i)
#            expect_rd_up,expect_rd_down = error_rdn(mbd,mbd_err,mdpls,mdpls_err,p,p_err,d,d_err,Vcb,Vcb_err,mmu,mmu_err,mtau,mtau_err,vev,vev_err,mc,mc_err,mb,mb_err,j,i)
            expect_rd = bsemi(mc,mb,mbd,mdpls,p,d,i,j)
            expect_rd_err = error_bsemi(mc,mc_err,mb,mb_err,mbd,mbd_err,mdpls,mdpls_err,p,p_err,d,d_err,i,j)
            expect_rd_up,expect_rd_down=expect_rd+expect_rd_err[0],expect_rd-expect_rd_err[1]
            mid_rd = 0.5*(expect_rd_up+expect_rd_down)
            mid_rds = 0.5*(expect_rds_up+expect_rds_down)
            sig_rd = sigma*(expect_rd_up-mid_rd)
            sig_rds = sigma*(expect_rds_up-mid_rds)
            rd_bool = ((av_rd >= mid_rd and mid_rd+sig_rd >= av_rd-sige_rd) or (av_rd <= mid_rd and mid_rd-sig_rd <= av_rd+sige_rd)) 
            rds_bool = ((av_rds >= mid_rds and mid_rds+sig_rds >= av_rds-sige_rds) or (av_rds <= mid_rds and mid_rds-sig_rds <= av_rds+sige_rds)) 

            #Oblique
            expect_SOblique = S2HDMofAlphaBeta(i,mA0,mH0,alph,b,mW,mZ,higgs,Gf,alp_EM,wangle)
            #theoretical error very small here
            S_up, S_down = SOb_err(i,mA0,mH0,alph,b,mW,mW_err,mZ,mZ_err,higgs,higgs_err,Gf,alp_EM,wangle,wangle_err)
            expect_SOblique_up,expect_SOblique_down = expect_SOblique+S_up,expect_SOblique-S_down
            mid_SOblique = 0.5*(expect_SOblique_up+expect_SOblique_down)
            sig_SOblique = sigma*(expect_SOblique_up-mid_SOblique)
            SOblique_bool=((av_SOblique>=mid_SOblique and mid_SOblique+sig_SOblique>=av_SOblique-sige_SOblique) or (av_SOblique<=mid_SOblique and mid_SOblique-sig_SOblique<=av_SOblique+sige_SOblique))

            expect_TOblique = T2HDMofAlphaBeta(i,mA0,mH0,alph,b,mW,mZ,higgs,Gf,alp_EM)
            #theoretical error very small here
            T_up, T_down = TOb_err(i,mA0,mH0,alph,b,mW,mW_err,mZ,mZ_err,higgs,higgs_err,Gf,alp_EM)
            expect_TOblique_up,expect_TOblique_down = expect_TOblique+T_up,expect_TOblique-T_down
            mid_TOblique = 0.5*(expect_TOblique_up+expect_TOblique_down)
            sig_TOblique = sigma*(expect_TOblique_up-mid_TOblique)
            TOblique_bool=((av_TOblique>=mid_TOblique and mid_TOblique>=av_TOblique-sige_TOblique) or (av_TOblique<=mid_TOblique and mid_TOblique<=av_TOblique+sige_TOblique))

            expect_UOblique = U2HDMofAlphaBeta(i,mA0,mH0,alph,b,mW,mZ,higgs,Gf,alp_EM,wangle)
            #theoretical error very small here
            U_up, U_down = UOb_err(i,mA0,mH0,alph,b,mW,mW_err,mZ,mZ_err,higgs,higgs_err,Gf,alp_EM,wangle,wangle_err)
            expect_UOblique_up,expect_UOblique_down = expect_UOblique+U_up,expect_UOblique-U_down
            mid_UOblique = 0.5*(expect_UOblique_up+expect_UOblique_down)
            sig_UOblique = sigma*(expect_UOblique_up-mid_UOblique)
            UOblique_bool=((av_UOblique>=mid_UOblique and mid_UOblique>=av_UOblique-sige_UOblique) or (av_UOblique<=mid_UOblique and mid_UOblique<=av_UOblique+sige_UOblique))

            ##### (SEMI-)LEPTONICS #####
            if bpls_bool and dpls_bool and dspls_bool and kpi_bool and tkpi_bool and bpmu_bool and dsmu_bool and rd_bool:# and rds_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mHl_loc = np.append(mHl_loc,i_log)
                tanbl_loc = np.append(tanbl_loc,j_log)
#                chi_lij = chisq_simp([av_b,av_d,av_ds,av_k,av_t,av_bm,av_dm,av_rd,av_rds],[mid_b,mid_d,mid_ds,mid_k,mid_t,mid_bmu,mid_dm,mid_rd,mid_rds],[sige_b,sige_d,sige_ds,sige_k,sige_t,sige_bm,sige_dm,sige_rd,sige_rds],[sig_b,sig_d,sig_ds,sig_k,sig_t,sig_bmu,sig_dm,sig_rd,sig_rds])
                chi_lij = chisq_simp([av_b,av_d,av_ds,av_k,av_t,av_bm,av_dm,av_rd],[mid_b,mid_d,mid_ds,mid_k,mid_t,mid_bmu,mid_dm,mid_rd],[sige_b,sige_d,sige_ds,sige_k,sige_t,sige_bm,sige_dm,sige_rd],[sig_b,sig_d,sig_ds,sig_k,sig_t,sig_bmu,sig_dm,sig_rd])
                chi_ls = np.append(chi_ls,chi_lij)
            
            ##### MIXING #####
            if bmix_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mHb_loc = np.append(mHb_loc,i_log)
                tanbb_loc = np.append(tanbb_loc,j_log)
                chi_mij = chisq_simp([av_bmix,av_bmixs],[mid_bm,mid_bms],[sige_bmix,sige_bmixs],[sig_bm,sig_bms])
                chi_ms = np.append(chi_ms,chi_mij)
               
            ##### BSGAMMA #####
            if gam_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mHg_loc = np.append(mHg_loc,i_log)
                tanbg_loc = np.append(tanbg_loc,j_log)
                chi_gij = chisq_simp([av_g],[mid_g],[sige_g],[sig_g])
                chi_gs = np.append(chi_gs,chi_gij)

            #Oblique
            if SOblique_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mHS_loc=np.append(mHS_loc,i_log)
                tanbS_loc=np.append(tanbS_loc,j_log)
                chi_Sij=chisq_simp([av_SOblique],[mid_SOblique],[sige_SOblique],[sig_SOblique])
                chi_Ss=np.append(chi_Ss,chi_Sij)

            if TOblique_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mHT_loc=np.append(mHT_loc,i_log)
                tanbT_loc=np.append(tanbT_loc,j_log)
                chi_Tij=chisq_simp([av_TOblique],[mid_TOblique],[sige_TOblique],[sig_TOblique])
                chi_Ts=np.append(chi_Ts,chi_Tij)
            
            if SOblique_bool and TOblique_bool and UOblique_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mHU_loc=np.append(mHU_loc,i_log)
                tanbU_loc=np.append(tanbU_loc,j_log)
                chi_Uij=chisq_simp([av_SOblique,av_TOblique,av_UOblique],[mid_SOblique,mid_TOblique,mid_UOblique],[sige_SOblique,sige_TOblique,sige_UOblique],[sig_SOblique,sig_TOblique,sig_UOblique])
                chi_Us=np.append(chi_Us,chi_Uij)

            ##### (SEMI-)LEPTONICS, MIXING, AND BSGAMMA #####
            if bpls_bool and dpls_bool and dspls_bool and bmix_bool and kpi_bool and tkpi_bool and gam_bool and bpmu_bool and dsmu_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mHa_loc = np.append(mHa_loc,i_log)
                tanba_loc = np.append(tanba_loc,j_log)
                chi_1ij = chisq_simp([av_b,av_d,av_ds,av_k,av_t,av_bmix,av_bmixs,av_g,av_bm,av_dm],[mid_b,mid_d,mid_ds,mid_k,mid_t,mid_bm,mid_bms,mid_g,mid_bmu,mid_dm],[sige_b,sige_d,sige_ds,sige_k,sige_t,sige_bmix,sige_bmixs,sige_g,sige_bm,sige_dm],[sig_b,sig_d,sig_ds,sig_k,sig_t,sig_bm,sig_bms,sig_g,sig_bmu,sig_dm])
                chi_1s = np.append(chi_1s,chi_1ij)
          
            ##### B TO MU MU #####
            if bs_bool and bd_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mHmu_loc = np.append(mHmu_loc,i_log)
                tanbmu_loc = np.append(tanbmu_loc,j_log)
                chi_uij = chisq_simp([av_bsmu,av_bdmu],[mid_bsmu,mid_bdmu],[sige_bsmu,sige_bdmu],[sig_bsmu,sig_bdmu])
                chi_mus = np.append(chi_mus,chi_uij)

            ##### GLOBAL #####
            if bpls_bool and dpls_bool and dspls_bool and bmix_bool and kpi_bool and tkpi_bool and gam_bool and bs_bool and bd_bool and rd_bool and bpmu_bool and dsmu_bool and SOblique_bool and TOblique_bool and UOblique_bool and rds_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mHa2_loc = np.append(mHa2_loc,i_log)
                tanba2_loc = np.append(tanba2_loc,j_log)
                chi_2ij = chisq_simp(
                        [av_b,av_d,av_ds,av_k,av_t,av_bmix,av_bmixs,av_g,av_bsmu,av_bdmu,av_rd,av_bm,av_dm, av_SOblique, av_TOblique, av_UOblique,av_rds],
                        [mid_b,mid_d,mid_ds,mid_k,mid_t,mid_bm,mid_bms,mid_g,mid_bsmu,mid_bdmu,mid_rd,mid_bmu,mid_dm,mid_SOblique,mid_TOblique,mid_UOblique,mid_rds],
                        [sige_b,sige_d,sige_ds,sige_k,sige_t,sige_bmix,sige_bmixs,sige_g,sige_bsmu,sige_bdmu,sige_rd,sige_bm,sige_dm,sige_SOblique,sige_TOblique,sige_UOblique,sige_rds],
                        [sig_b,sig_d,sig_ds,sig_k,sig_t,sig_bm,sig_bms,sig_g,sig_bsmu,sig_bdmu,sig_rd,sig_bmu,sig_dm,sig_SOblique,sig_TOblique,sig_UOblique,sig_rds])
                chi_2s = np.append(chi_2s,chi_2ij)
                if chi_2ij < chi_2min[0]:
                    chi_2min = [chi_2ij,i,j]

    return [mHl_loc, mHb_loc, mHg_loc, mHa_loc, mHmu_loc, mHa2_loc,mHS_loc,mHT_loc,mHU_loc], [tanbl_loc, tanbb_loc, tanbg_loc, tanba_loc, tanbmu_loc, tanba2_loc, tanbS_loc,tanbT_loc,tanbU_loc], [chi_ls, chi_ms, chi_gs, chi_1s, chi_mus, chi_2s,chi_Ss,chi_Ts,chi_Us], chi_2min
