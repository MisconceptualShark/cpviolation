from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from functions import *

g_gev = (1.1663787e-5)**2
Gf = 1.1663787e-5
hbar_gev = 6.582119514e-25

def tdfits(
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
        something here if needed
    '''
    sigma = 3

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

    chis,chi_min = [],[100,0,0,0]
    log_mH = np.linspace(2,3.5,75)
    log_tanb = np.linspace(-1,2,150)
    log_mA = np.linspace(2,3.5,75)
    mHs,tanbs,mAs = 10**log_mH,10**log_tanb,10**log_mA
    mH_loc,tanb_loc,mA_loc = [],[],[]

    for i in mHs:
        for j in tanbs:
            for k in mAs:
                b = np.arctan(j)
                alph = b - np.pi/2

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
                expect_bs = bmumu(mt,tbs,fBs,Vtb,Vts,mmu,mbs,mW,j,i,mb,ms,mc,mu,wangle,higgs,vev,Vus,Vub,Vcs,Vcb,k,alph,lam_QCD)
                expect_bs_uperr,expect_bs_downerr = error_bmumu(mt,mt_err,tbs,tbs_err,fBs,fBs_err,Vtb,Vtb_err,Vts,Vts_err,mmu,mmu_err,mbs,mbs_err,mW,mW_err,j,i,mb,mb_err,ms,ms_err,mc,mc_err,mu,mu_err,wangle,wangle_err,higgs,higgs_err,vev,vev_err,Vus,Vus_err,Vub,Vub_err,Vcs,Vcs_err,Vcb,Vcb_err,k,alph,lam_QCD,QCD_err)
                expect_bs_up, expect_bs_down = expect_bs+expect_bs_uperr, expect_bs-expect_bs_downerr
                mid_bsmu=0.5*(expect_bs_up+expect_bs_down)
                sig_bsmu=sigma*(expect_bs_up-mid_bsmu)
                bs_bool = ((av_bsmu >= mid_bsmu and mid_bsmu+sig_bsmu >= av_bsmu-sige_bsmu) or (av_bsmu <= mid_bsmu and mid_bsmu-sig_bsmu <= av_bsmu+sige_bsmu))

                expect_bd = bmumu(mt,tbd,fbpls,Vtb,Vtd,mmu,mbd,mW,j,i,mb,ms,mc,mu,wangle,higgs,vev,Vus,Vub,Vcs,Vcb,k,alph,lam_QCD)
                expect_bd_uperr,expect_bd_downerr = error_bmumu(mt,mt_err,tbd,tbd_err,fbpls,fbpls_err,Vtb,Vtb_err,Vtd,Vtd_err,mmu,mmu_err,mbd,mbd_err,mW,mW_err,j,i,mb,mb_err,ms,ms_err,mc,mc_err,mu,mu_err,wangle,wangle_err,higgs,higgs_err,vev,vev_err,Vus,Vus_err,Vub,Vub_err,Vcs,Vcs_err,Vcb,Vcb_err,k,alph,lam_QCD,QCD_err)
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
                expect_SOblique=S2HDMofAlphaBeta (i,k,k,alph,np.arctan(j),mW,mZ,higgs,Gf,alp_EM,wangle)
                #theoretical error very small here
                sig_SOblique=0
                expect_SOblique_up,expect_SOblique_down=expect_SOblique,expect_SOblique
                mid_SOblique=expect_SOblique
                SOblique_bool=((av_SOblique>=mid_SOblique and mid_SOblique>=av_SOblique-sige_SOblique) or (av_SOblique<=mid_SOblique and mid_SOblique<=av_SOblique+sige_SOblique))

                expect_TOblique=T2HDMofAlphaBeta (i,k,k,alph,np.arctan(j),mW,mZ,higgs,Gf,alp_EM)
                #theoretical error very small here
                sig_TOblique=0
                expect_TOblique_up,expect_TOblique_down=expect_TOblique,expect_TOblique
                mid_TOblique=expect_TOblique
                TOblique_bool=((av_TOblique>=mid_TOblique and mid_TOblique>=av_TOblique-sige_TOblique) or (av_TOblique<=mid_TOblique and mid_TOblique<=av_TOblique+sige_TOblique))

                expect_UOblique=U2HDMofAlphaBeta (i,k,k,alph,np.arctan(j),mW,mZ,higgs,Gf,alp_EM,wangle)
                #theoretical error very small here
                sig_UOblique=0
                expect_UOblique_up,expect_UOblique_down=expect_UOblique,expect_UOblique
                mid_UOblique=expect_UOblique
                UOblique_bool=((av_UOblique>=mid_UOblique and mid_UOblique>=av_UOblique-sige_UOblique) or (av_UOblique<=mid_UOblique and mid_UOblique<=av_UOblique+sige_UOblique))

                if bpls_bool and dpls_bool and dspls_bool and bmix_bool and kpi_bool and tkpi_bool and gam_bool and bs_bool and bd_bool and rd_bool and bpmu_bool and dsmu_bool and SOblique_bool and TOblique_bool and UOblique_bool and rds_bool:
                    i_log, j_log, k_log = np.log10(i),np.log10(j),np.log10(k)
                    mH_loc = np.append(mH_loc,i_log)
                    tanb_loc = np.append(tanb_loc,j_log)
                    mA_loc = np.append(mA_loc,k_log)
                    chit = chisq_simp(
                            [av_b,av_d,av_ds,av_k,av_t,av_bmix,av_bmixs,av_g,av_bsmu,av_bdmu,av_rd,av_bm,av_dm, av_SOblique, av_TOblique, av_UOblique,av_rds],
                            [mid_b,mid_d,mid_ds,mid_k,mid_t,mid_bm,mid_bms,mid_g,mid_bsmu,mid_bdmu,mid_rd,mid_bmu,mid_dm,mid_SOblique,mid_TOblique,mid_UOblique,mid_rds],
                            [sige_b,sige_d,sige_ds,sige_k,sige_t,sige_bmix,sige_bmixs,sige_g,sige_bsmu,sige_bdmu,sige_rd,sige_bm,sige_dm,sige_SOblique,sige_TOblique,sige_UOblique,sige_rds],
                            [sig_b,sig_d,sig_ds,sig_k,sig_t,sig_bm,sig_bms,sig_g,sig_bsmu,sig_bdmu,sig_rd,sig_bmu,sig_dm,sig_SOblique,sig_TOblique,sig_UOblique,sig_rds])
                    chis = np.append(chis,chit)
                    if chit < chi_min[0]:
                        chi_min = [chit,i,j,k]

    return mH_loc, tanb_loc, mA_loc, chis, chi_min
