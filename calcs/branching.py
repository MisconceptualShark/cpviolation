from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

g = (1.1663787e-5)**2
hbar = 6.58212e-25

def bsm(mm,ml,Vud,fm,taum):
    '''
        Calculates SM branching ratio 
    '''
    Bs = (1/(8*np.pi))*(g*mm*ml**2)*((1-(ml**2/mm**2))**2)*(Vud**2)*(fm**2)*taum
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
#            gam_bet = exp_branch_up - expect_branch_down
#            alp_delt = expect_branch_down - exp_branch_down
#            if alp_delt/gam_bet >= 0.95 or gam_bet/alp_delt >= 0.95:
            if (branch_exp >= expect_branch and expect_branch_up >= exp_branch_down) or (branch_exp <= expect_branch and expect_branch_down <= exp_branch_up):
                i_log, j_log = np.log10(i), np.log10(j)
                mH_loc = np.append(mH_loc,i_log)
                tanb_loc = np.append(tanb_loc,j_log)

    return mH_loc, tanb_loc

def higgsy(bexp,mm,mu,md,tanb,ml,Vud,fm,taum):
    '''
        Function of tan(beta) and other stuff for higgs mass
    '''
    bs = bsm(mm,ml,Vud,fm,taum)
    rh_plus = -1 + np.sqrt(bexp/bs)
    rh_min = -1 - np.sqrt(bexp/bs)
    delta_plus = (mm**2/rh_plus)*(mu - md*tanb**2)/(mu + md)
    delta_min = (mm**2/rh_min)*(mu - md*tanb**2)/(mu + md)
    imagine_plus = []
    imagine_min = []
    for i in range(len(delta_plus)):
        if delta_plus[i] < 0:
            imagine_plus = np.append(imagine_plus,i)
    delta_plus = np.delete(delta_plus,imagine_plus)
    for i in range(len(delta_min)):
        if delta_min[i] < 0:
            imagine_min = np.append(imagine_min,i)
    delta_min = np.delete(delta_min,imagine_min)
    higgs_plus = np.sqrt(delta_plus)
    higgs_min = np.sqrt(delta_min)
    higgs = np.append(higgs_plus,higgs_min)
    return higgs
        
def errors_2(bexp,bexp_err,mm,mm_err,mu,mu_err,md,md_err,tanb,ml,ml_err,Vud,Vud_err,fm,fm_err,taum,taum_err):
    '''
        probabilistic boogaloo
    '''
    err1_up = abs(higgsy(bexp+bexp_err[0],mm,mu,md,tanb,ml,Vud,fm,taum)-higgsy(bexp,mm,mu,md,tanb,ml,Vud,fm,taum))
    err1_down = abs(higgsy(bexp+bexp_err[1],mm,mu,md,tanb,ml,Vud,fm,taum)-higgsy(bexp,mm,mu,md,tanb,ml,Vud,fm,taum))
    err2_up = abs(higgsy(bexp,mm+mm_err[0],mu,md,tanb,ml,Vud,fm,taum)-higgsy(bexp,mm,mu,md,tanb,ml,Vud,fm,taum))
    err2_down = abs(higgsy(bexp,mm+mm_err[1],mu,md,tanb,ml,Vud,fm,taum)-higgsy(bexp,mm,mu,md,tanb,ml,Vud,fm,taum))
    err3_up = abs(higgsy(bexp,mm,mu,md+md_err[0],tanb,ml,Vud,fm,taum)-higgsy(bexp,mm,mu,md,tanb,ml,Vud,fm,taum))
    err3_down = abs(higgsy(bexp,mm,mu,md+md_err[1],tanb,ml,Vud,fm,taum)-higgsy(bexp,mm,mu,md,tanb,ml,Vud,fm,taum))
    err4_up = abs(higgsy(bexp,mm,mu+mu_err[0],md,tanb,ml,Vud,fm,taum)-higgsy(bexp,mm,mu,md,tanb,ml,Vud,fm,taum))
    err4_down = abs(higgsy(bexp,mm,mu+mu_err[1],md,tanb,ml,Vud,fm,taum)-higgsy(bexp,mm,mu,md,tanb,ml,Vud,fm,taum))
    err5_up = abs(higgsy(bexp,mm,mu,md,tanb,ml+ml_err[0],Vud,fm,taum)-higgsy(bexp,mm,mu,md,tanb,ml,Vud,fm,taum))
    err5_down = abs(higgsy(bexp,mm,mu,md,tanb,ml+ml_err[1],Vud,fm,taum)-higgsy(bexp,mm,mu,md,tanb,ml,Vud,fm,taum))
    err6_up = abs(higgsy(bexp,mm,mu,md,tanb,ml,Vud+Vud_err[0],fm,taum)-higgsy(bexp,mm,mu,md,tanb,ml,Vud,fm,taum))
    err6_down = abs(higgsy(bexp,mm,mu,md,tanb,ml,Vud+Vud_err[1],fm,taum)-higgsy(bexp,mm,mu,md,tanb,ml,Vud,fm,taum))
    err7_up = abs(higgsy(bexp,mm,mu,md,tanb,ml,Vud,fm+fm_err[0],taum)-higgsy(bexp,mm,mu,md,tanb,ml,Vud,fm,taum))
    err7_down = abs(higgsy(bexp,mm,mu,md,tanb,ml,Vud,fm+fm_err[1],taum)-higgsy(bexp,mm,mu,md,tanb,ml,Vud,fm,taum))
    err8_up = abs(higgsy(bexp,mm,mu,md,tanb,ml,Vud,fm,taum+taum_err[0])-higgsy(bexp,mm,mu,md,tanb,ml,Vud,fm,taum))
    err8_down = abs(higgsy(bexp,mm,mu,md,tanb,ml,Vud,fm,taum+taum_err[1])-higgsy(bexp,mm,mu,md,tanb,ml,Vud,fm,taum))

    upper = np.sqrt(err1_up**2 + err2_up**2 + err3_up**2 + err4_up**2 + err5_up**2 + err6_up**2 + err7_up**2 + err8_up**2)
    lower = np.sqrt(err1_down**2 + err2_down**2 + err3_down**2 + err4_down**2 + err5_down**2 + err6_down**2 + err7_down**2 + err8_down**2)
    higgs_up = higgsy(bexp,mm,mu,md,tanb,ml,Vud,fm,taum) + upper
    higgs_down = higgsy(bexp,mm,mu,md,tanb,ml,Vud,fm,taum) - lower

    return higgs_up, higgs_down

m_bplus, m_bplus_err = [5.27925,[0.00026,-0.00026]]
m_dplus, m_dplus_err = [1.8695,[0.0004,-0.0004]]
m_dsplus, m_dsplus_err = [1.969,[0.0014,-0.0014]]
m_tau, m_tau_err = [1.77686,[0.00012,-0.00012]]
m_mu, m_mu_err = [0.1056583745,[0.0000000024,-0.0000000024]]
m_u, m_u_err = [0.00216,[0.00049,-0.00026]]
m_d, m_d_err = [0.00467,[0.00048,-0.00017]]
m_c, m_c_err = [1.27,[0.02,-0.02]]
m_b, m_b_err = [4.18,[0.03,-0.02]]
m_s, m_s_err = [0.093,[0.011,-0.005]]

Vub, Vub_err = [0.00375,[0.0003,-0.00021]]
Vcd, Vcd_err = [0.22461,[0.00106,-0.00018]]
Vcs, Vcs_err = [0.97353,[0.0001,-0.00025]]

f_bplus, f_bplus_err = [0.190,[0.0013,-0.0013]]
f_dplus, f_dplus_err = [0.212,[0.0007,-0.0007]]
f_dsplus, f_dsplus_err = [0.2499,[0.0005,-0.0005]]

tau_bplus, tau_bplus_err = [(1.638e-12)/hbar,[(0.004e-12)/hbar,-(0.004e-12)/hbar]]
tau_dplus, tau_dplus_err = [(1040e-15)/hbar,[(7e-15)/hbar,-(7e-15)/hbar]]
tau_dsplus, tau_dsplus_err = [(504e-15)/hbar,[(4e-15)/hbar,-(4e-15)/hbar]]

bplus_exp, bplus_err_exp = [0.847e-4,[0.150e-4,-0.097e-4]]
dplus_exp, dplus_err_exp = [4.019e-4,[0.11e-4,-0.15e-4]]
dsplus_exp, dsplus_err_exp = [0.05107,[0.0011,-0.0013]]

#br = bsm(m_dplus,m_mu,Vcd,f_dplus,tau_dplus)
#err_up, err_down = error_branching(m_dplus,m_dplus_err,m_mu,m_mu_err,Vcd,Vcd_err,f_dplus,f_dplus_err,tau_dplus,tau_dplus_err)
#print err_up, err_down
#log_tanb = np.linspace(-1,2,300)
#tanb = 10**log_tanb
#b_up, b_down = errors_2(bplus_exp,bplus_err_exp,m_bplus,m_bplus_err,m_u,m_u_err,m_b,m_b_err,tanb,m_tau,m_tau_err,Vub,Vub_err,f_bplus,f_bplus_err,tau_bplus,tau_bplus_err)
#tanb = np.append(tanb,tanb)

#plt.figure()
#plt.plot(tanb,b_up)
#plt.plot(tanb,b_down)
#plt.plot(tanb,b_up-b_down)
#plt.fill_between(tanb,b_down,b_up,color='gray')
#plt.yscale('log')
#plt.xscale('log')
#plt.show()

mH_bplus, tanb_bplus = itera(m_bplus,m_bplus_err,m_tau,m_tau_err,Vub,Vub_err,f_bplus,f_bplus_err,tau_bplus,tau_bplus_err,m_u,m_u_err,m_b,m_b_err,bplus_exp,bplus_err_exp)

plt.figure()
plt.scatter(tanb_bplus,mH_bplus,c='green',marker=',')
plt.ylabel('$\\log[m_{H+}$, GeV]')
plt.xlabel('$\\log[\\tan(\\beta)]$')
plt.title('$B^+\\to\\tau^+\\nu$')

#expect, err_up, err_down, rd = itera(m_dplus,m_dplus_err,m_mu,m_mu_err,Vcd,Vcd_err,f_dplus,f_dplus_err,tau_dplus,tau_dplus_err,m_c,m_c_err,m_d,m_d_err,dplus_exp,dplus_err_exp)
#fig = plt.figure()
#plt.plot(rd, expect
mH_dplus, tanb_dplus = itera(m_dplus,m_dplus_err,m_mu,m_mu_err,Vcd,Vcd_err,f_dplus,f_dplus_err,tau_dplus,tau_dplus_err,m_c,m_c_err,m_d,m_d_err,dplus_exp,dplus_err_exp)

plt.figure()
plt.scatter(tanb_dplus,mH_dplus,c='green',marker=',')
plt.ylabel('$\\log[m_{H+}$, GeV]')
plt.xlabel('$\\log[\\tan(\\beta)]$')
plt.title('$D^+\\to\\mu^+\\nu$')
#
mH_dsplus, tanb_dsplus = itera(m_dsplus,m_dsplus_err,m_tau,m_tau_err,Vcs,Vcs_err,f_dsplus,f_dsplus_err,tau_dsplus,tau_dsplus_err,m_c,m_c_err,m_s,m_s_err,dsplus_exp,dsplus_err_exp)

plt.figure()
plt.scatter(tanb_dsplus,mH_dsplus,c='green',marker=',')
plt.ylabel('$\\log[m_{H+}$, GeV]')
plt.xlabel('$\\log[\\tan(\\beta)]$')
plt.title('$D_s^+\\to\\tau^+\\nu$')
plt.show()




