from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

g2 = (1.1663788e-11)**2
hbar = 6.582119e-22
#hbar = 1

def mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq):
    '''
        B mixing mass eqn
    '''
    x_tH = mt**2/mH**2
    x_tW = mt**2/mW**2
    x_HW = mH**2/mW**2
    S_WH = (x_tH/tanb**2)*((2*x_HW-8)*np.log(x_tH)/((1-x_HW)*(1-x_tH)**2) + 6*x_HW*np.log(x_tW)/((1-x_HW)*(1-x_tW)**2) - (8-2*x_tW)/((1-x_tW)*(1-x_tH)))
    S_WW = (1 + 9/(1-x_tW) - 6/((1-x_tW)**2) - 6*(x_tW**2)*np.log(x_tW)/((1-x_tW)**3))
    S_HH = (x_tH/tanb**4)*((1+x_tH)/((1-x_tH)**2) + 2*x_tH*np.log(x_tH)/((1-x_tH)**3))

    delt_mq = g2/(24*np.pi**2)*((Vtq*Vtb)**2)*etaB*mB*(mt**2)*(fBq**2)*BBq*(S_WW + S_WH + S_HH)

    return delt_mq/hbar

def error_branching(mt,mt_err,mH,mW,mW_err,tanb,Vtq,Vtq_err,Vtb,Vtb_err,etaB,mB,mB_err,fBq,fBq_err,BBq,BBq_err):
    '''
        Calculates errors in branching ratios, using functional method
        - all err vars are [up,low] 
    '''
    mix = mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq)
    err1_up = abs(mixing(mt+mt_err[0],mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq)-mix)
    err1_low = abs(mixing(mt+mt_err[1],mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq)-mix)
    err2_up = abs(mixing(mt,mH,mW+mW_err[0],tanb,Vtq,Vtb,etaB,mB,fBq,BBq)-mix)
    err2_low = abs(mixing(mt,mH,mW+mW_err[1],tanb,Vtq,Vtb,etaB,mB,fBq,BBq)-mix)
    err3_up = abs(mixing(mt,mH,mW,tanb,Vtq+Vtq_err[0],Vtb,etaB,mB,fBq,BBq)-mix)
    err3_low = abs(mixing(mt,mH,mW,tanb,Vtq+Vtq_err[1],Vtb,etaB,mB,fBq,BBq)-mix)
    err4_up = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb+Vtb_err[0],etaB,mB,fBq,BBq)-mix)
    err4_low = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb+Vtb_err[1],etaB,mB,fBq,BBq)-mix)
    err6_up = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB+mB_err[0],fBq,BBq)-mix)
    err6_low = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB+mB_err[1],fBq,BBq)-mix)
    err7_up = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq+fBq_err[0],BBq)-mix)
    err7_low = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq+fBq_err[1],BBq)-mix)
    err8_up = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq+BBq_err[0])-mix)
    err8_low = abs(mixing(mt,mH,mW,tanb,Vtq,Vtb,etaB,mB,fBq,BBq+BBq_err[1])-mix)

    upper = np.sqrt(err1_up**2 + err2_up**2 + err3_up**2 + err4_up**2 + err6_up**2 + err7_up**2 + err8_up**2)
    lower = np.sqrt(err1_low**2 + err2_low**2 + err3_low**2 + err4_low**2 + err6_low**2 + err7_low**2 + err7_low**2)

    return upper, lower

def itera(mt,mt_err,mW,mW_err,Vtq,Vtq_err,Vtb,Vtb_err,etaB,mB,mB_err,fBq,fBq_err,BBq,BBq_err,branch_exp,branch_exp_error):
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
            expect_branch = mixing(mt,i,mW,j,Vtq,Vtb,etaB,mB,fBq,BBq)
            print expect_branch
            expect_error = error_branching(mt,mt_err,i,mW,mW_err,j,Vtq,Vtq_err,Vtb,Vtb_err,etaB,mB,mB_err,fBq,fBq_err,BBq,BBq_err)
            expect_branch_up, expect_branch_down = expect_branch+expect_error[0],expect_branch-expect_error[1]
            if (branch_exp > expect_branch and expect_branch_up > exp_branch_down) or (branch_exp < expect_branch and expect_branch_down < exp_branch_up):
                i_log, j_log = np.log10(i), np.log10(j)
                mH_loc = np.append(mH_loc,i_log)
                tanb_loc = np.append(tanb_loc,j_log)

    return mH_loc, tanb_loc

mt, mt_err = [172.9e3,[0.4e3,-0.4e3]]
mW, mW_err = [80.379e3,[0.012e3,-0.012e3]]
mBd, mBd_err = [5279.55,[0.26,-0.26]]
mBs, mBs_err = [5366.84,[0.21,-0.21]]

Vts, Vts_err = [0.04169,[0.00082,-0.00212]]
Vtd, Vtd_err = [0.00871,[0.00025,-0.00065]]
Vtb, Vtb_err = [0.999093,[0.00009,-0.000036]]

alpha_s, alpha_s_err = [0.1181,[0.0011,-0.0011]]
etaB = 0.4942

fBs, fBs_err = [228.4,[3.7,-3.7]]
fBd, fBd_err = [190.5,[1.3,-1.3]]

BBs, BBs_err = [1.35,[0.06,-0.06]]
BBd, BBd_err = [1.3,[0.1,-0.1]]

delt_md, delt_md_err = [0.5064e12,[0.0019e12,-0.0019e12]]
delt_ms, delt_ms_err = [17.757e12,[0.021e12,-0.021e12]]

mH_md, tanb_md = itera(mt,mt_err,mW,mW_err,Vtd,Vtd_err,Vtb,Vtb_err,etaB,mBd,mBd_err,fBd,fBd_err,BBd,BBd_err,delt_md,delt_md_err)

plt.figure()
plt.scatter(tanb_md,mH_md,marker=',',c='green')
plt.axis([-1,2,3,6])
plt.ylabel('$\\log[m_{H+}$, MeV]')
plt.xlabel('$\\log[\\tan(\\beta)]$')
plt.title('$B^0_d-\\bar{B}^0_d$')

mH_ms, tanb_ms = itera(mt,mt_err,mW,mW_err,Vts,Vts_err,Vtb,Vtb_err,etaB,mBs,mBs_err,fBs,fBs_err,BBs,BBs_err,delt_ms,delt_ms_err)

plt.figure()
plt.scatter(tanb_ms,mH_ms,marker=',',c='green')
plt.axis([-1,2,3,6])
plt.ylabel('$\\log[m_{H+}$, MeV]')
plt.xlabel('$\\log[\\tan(\\beta)]$')
plt.title('$B^0_s-\\bar{B}^0_s$')

plt.show()
