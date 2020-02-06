from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def exp_like(err,x,xt):
    like = 1
    for i in range(len(x)):
        like_i = (1/(np.sqrt(2*np.pi)*err[i]))*np.exp(-0.5*((x[i]-xt[i])/err[i])**2)
        like = like*like_i
    return like

def cov(x,y):
    xbar, ybar = x.mean(), y.mean()
    return np.sum((x-xbar)*(y-ybar))/(len(x)-1)

def cov_mat(X):
    return np.array([[cov(X[0],X[0]),cov(X[0],X[1])], \
                     [cov(X[1],X[0]),cov(X[1],X[1])]])

def chisq_simp(obs,the,sige,sigt):
    '''
        chisq val, all parameters lists of values, simple
    '''
    N = len(obs)
    chi = 0
    for i in range(N):
        sig = np.sqrt(sige[i]**2 + sigt[i]**2)
        val = (obs[i]-the[i])/sig
        chi += val**2
    return chi

def chisq_full(bpls_exp,bpls_exp_error,dpls_exp,dpls_exp_error,dspls_exp,dspls_exp_error,bmix_exp,bmix_exp_error,bmix_sm,bmix_sm_error,kpi_exp,kpi_exp_error,tkpi_exp,tkpi_exp_error,gams_exp,gams_exp_error,gamc_exp,gamc_exp_error,mbpls,mbpls_err,mdpls,mdpls_err,mdspls,mdspls_err,mK,mK_err,mpi,mpi_err,mB,mB_err,mtau,mtau_err,mmu,mmu_err,etaB,etaB_err,fBd,fBd_err,Bbd,Bbd_err,fbpls,fbpls_err,fdpls,fdpls_err,fdspls,fdspls_err,fKpi,fKpi_err,delt_kpi,delt_kpi_err,delt_tau,delt_tau_err,tbpls,tbpls_err,tdpls,tdpls_err,tdspls,tdspls_err,mu,mu_err,md,md_err,mc,mc_err,ms,ms_err,mb,mb_err,mt,mt_err,mtb,mtb_err,mW,mW_err,mWb,mWb_err,mub,lam_QCD,QCD_err,hi,a,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamu,gamu_err,alp_EM,Vud,Vud_err,Vus,Vus_err,Vub,Vub_err,Vcd,Vcd_err,Vcs,Vcs_err,Vcb,Vcb_err,Vtd,Vtd_err,Vts,Vts_err,Vtb,Vtb_err,tbd,tbd_err,tbs,tbs_err,fBs,fBs_err,mbd,mbd_err,mbs,mbs_err,bs_e,bs_eerr,bd_e,bd_eerr,hs,ts):
    '''
        chisq val, probably more in detail for different theories
    '''
    gam_exp = gams_exp/gamc_exp
    xgam = gam_exp*np.sqrt((gamc_exp_error[0]/gamc_exp)**2 + (gams_exp_error[0]/gams_exp)**2)
    ygam = gam_exp*np.sqrt((gamc_exp_error[1]/gamc_exp)**2 + (gams_exp_error[1]/gams_exp)**2)
    hs2 = 10**hs
    ts2 = 10**ts
    for i in hs2:
        for j in ts2:
            bpls_the, dpls_the, dspls_the = bthe(mbpls,mtau,Vub,fbpls,tbpls,mu,mb,j,i),bthe(mdpls,mmu,Vcd,fdpls,tdpls,mc,md,j,i),bthe(mdspls,mtau,Vcs,fdspls,tdspls,mc,ms,j,i)
            bpls_err, dpls_err, dspls_err = error_branching(mbpls,mbpls_err,mtau,mtau_err,Vub,Vub_err,fbpls,fbpls_err,tbpls,tbpls_err,mu,mu_err,mb,mb_err,j,i),error_branching(mdpls,mdpls_err,mmu,mmu_err,Vcd,Vcd_err,fdpls,fdpls_err,tdpls,tdpls_err,mc,mc_err,md,md_err,j,i),error_branching(mdspls,mdspls_err,mtau,mtau_err,Vcs,Vcs_err,fdspls,fdspls_err,tdspls,tdspls_err,mc,mc_err,ms,ms_err,j,i)
            b_chi = pow((bpls_exp-bpls_the)/np.sqrt(bpls_exp_error**2 + bpls_err**2),2)
            d_chi = pow((dpls_exp-dpls_the)/np.sqrt(dpls_exp_error**2 + dpls_err**2),2)
            ds_chi = pow((dspls_exp-dspls_the)/np.sqrt(dspls_exp_error**2 + dspls_err**2),2)

            bmix_the = mixing(mtb,i*1e3,mWb,j,Vtd,Vtb,etaB,mB,fBd,Bbd,bmix_sm)
            bmix_err = error_mixing(mtb,mtb_err,i*1e3,mWb,mWb_err,j,Vtd,Vtd_err,Vtb,Vtb_err,etaB,etaB_err,mB,mB_err,fBd,fBd_err,Bbd,Bbd_err,bmix_sm,bmix_sm_error)
            bmix_chi = pow((bmix_exp-bmix_the)/np.sqrt(bmix_exp_error**2 + bmix_err**2),2)

            kpi_the,tkpi_the = decay_bsm(mK,mpi,mmu,mtau,Vus,Vud,fKpi,delt_kpi,delt_tau,ms,md,mu,j,i)
            kpi_uperr,kpi_derr,tkpi_uperr,tkpi_derr = error_kpi(mK,mK_err,mpi,mpi_err,mmu,mmu_err,mtau,mtau_err,Vus,Vus_err,Vud,Vud_err,fKpi,fKpi_err,delt_kpi,delt_kpi_err,delt_tau,delt_tau_err,ms,ms_err,md,md_err,mu,mu_err,j,i)
            kpi_chi = pow((kpi_exp-kpi_the)/np.sqrt(kpi_exp_error**2 + kpi_err**2),2)
            tkpi_chi = pow((tkpi_exp-tkpi_the)/np.sqrt(tkpi_exp_error**2 + tkpi_err**2),2)

            gam_the = bsgamma(mt,mW,mub,lam_QCD,hi,a,i,j,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc_exp,gamu,Vub,Vts,Vtb,Vcb,alp_EM)
            gam_err = error_gamma(mt,mt_err,mW,mW_err,mub,lam_QCD,QCD_err,hi,a,i,j,A0,ac,at,a_s,B0,bc,bt,bs,delt_mc,delt_mt,delt_as,gamc_exp,gamc_exp_error,gamu,gamu_err,Vub,Vub_err,Vts,Vts_err,Vtb,Vtb_err,Vcb,Vcb_err,alp_EM)
            tkpi_chi = pow((tkpi_exp-tkpi_the)/np.sqrt(tkpi_exp_error**2 + tkpi_err**2),2)

            expect_bd,expect_bs = bmumu(mt,tbd,tbs,fBd,fBs,Vtd,Vts,mmu,mbd,mbs,mW,j,i)
            expect_bd_uperr,expect_bd_downerr,expect_bs_uperr,expect_bs_downerr = error_bmumu(mt,mt_err,tbd,tbd_err,tbs,tbs_err,fBd,fBd_err,fBs,fBs_err,Vtd,Vtd_err,Vts,Vts_err,mmu,mmu_err,mbd,mbd_err,mbs,mbs_err,mW,mW_err,j,i)

            chisq = 

#finish, asymmetric errors?




