from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from functions import *

g_gev = (1.1663787e-5)**2
Gf = 1.1663787e-5
hbar_gev = 6.582119514e-25

def tdfits(
        bsmu_exp,bsmu_exp_err,bdmu_exp,bdmu_exp_err,
        mu,mu_err,md,md_err,mc,mc_err,ms,ms_err,mb,mb_err,mt,mt_err,mW,mW_err,
        mbs,mbs_err,mbd,mbd_err,mmu,mmu_err,
        Vud,Vud_err,Vus,Vus_err,Vub,Vub_err,Vcd,Vcd_err,Vcs,Vcs_err,Vcb,Vcb_err,Vtd,Vtd_err,Vts,Vts_err,Vtb,Vtb_err,
        fBs,fBs_err,fBd,fBd_err,tbs,tbs_err,tbd,tbd_err,alp_EM,
        wangle,wangle_err,lam_QCD,QCD_err,higgs,higgs_err,vev,vev_err,
        SOblique,SOblique_err,TOblique,TOblique_err,UOblique,UOblique_err,mZ,mZ_err):
    '''
        something here if needed
    '''
    sigma = 1.96

    bsmu_exp_up,bsmu_exp_down = bsmu_exp+bsmu_exp_err[0],bsmu_exp+bsmu_exp_err[1]
    av_bsmu = 0.5*(bsmu_exp_up+bsmu_exp_down)
    sige_bsmu = sigma*(bsmu_exp_up-av_bsmu)

    bdmu_exp_up,bdmu_exp_down = bdmu_exp+bdmu_exp_err[0],bdmu_exp+bdmu_exp_err[1]
    av_bdmu = 0.5*(bdmu_exp_up+bdmu_exp_down)
    sige_bdmu = sigma*(bdmu_exp_up-av_bdmu)

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
                alph = b - np.arccos(0.465)

                expect_bs = bmumu(mt,tbs,fBs,Vtb,Vts,mmu,mbs,mW,j,i,mb,ms,mc,mu,wangle,higgs,vev,Vus,Vub,Vcs,Vcb,k,alph,lam_QCD)
                expect_bs_uperr,expect_bs_downerr = error_bmumu(mt,mt_err,tbs,tbs_err,fBs,fBs_err,Vtb,Vtb_err,Vts,Vts_err,mmu,mmu_err,mbs,mbs_err,mW,mW_err,j,i,mb,mb_err,ms,ms_err,mc,mc_err,mu,mu_err,wangle,wangle_err,higgs,higgs_err,vev,vev_err,Vus,Vus_err,Vub,Vub_err,Vcs,Vcs_err,Vcb,Vcb_err,k,alph,lam_QCD,QCD_err)
                expect_bs_up, expect_bs_down = expect_bs+expect_bs_uperr, expect_bs-expect_bs_downerr
                mid_bsmu=0.5*(expect_bs_up+expect_bs_down)
                sig_bsmu=sigma*(expect_bs_up-mid_bsmu)
                bs_bool = ((av_bsmu >= mid_bsmu and mid_bsmu+sig_bsmu >= av_bsmu-sige_bsmu) or (av_bsmu <= mid_bsmu and mid_bsmu-sig_bsmu <= av_bsmu+sige_bsmu)) 

                expect_bd = bmumu(mt,tbd,fBd,Vtb,Vtd,mmu,mbd,mW,j,i,mb,ms,mc,mu,wangle,higgs,vev,Vus,Vub,Vcs,Vcb,k,alph,lam_QCD)
                expect_bd_uperr,expect_bd_downerr = error_bmumu(mt,mt_err,tbd,tbd_err,fBd,fBd_err,Vtb,Vtb_err,Vtd,Vtd_err,mmu,mmu_err,mbd,mbd_err,mW,mW_err,j,i,mb,mb_err,ms,ms_err,mc,mc_err,mu,mu_err,wangle,wangle_err,higgs,higgs_err,vev,vev_err,Vus,Vus_err,Vub,Vub_err,Vcs,Vcs_err,Vcb,Vcb_err,k,alph,lam_QCD,QCD_err)
                expect_bd_up, expect_bd_down = expect_bd+expect_bd_uperr, expect_bd-expect_bd_downerr
                mid_bdmu=0.5*(expect_bd_up+expect_bd_down)
                sig_bdmu=sigma*(expect_bd_up-mid_bdmu)
                bd_bool = ((av_bdmu >= mid_bdmu and mid_bdmu+sig_bdmu >= av_bdmu-sige_bdmu) or (av_bdmu <= mid_bdmu and mid_bdmu-sig_bdmu <= av_bdmu+sige_bdmu)) 

                expect_SOblique=S2HDMofAlphaBeta(i,k,k,alph,np.arctan(j),mW,mZ,higgs,Gf,alp_EM,wangle)
                #theoretical error very small here
                sig_SOblique=0
                expect_SOblique_up,expect_SOblique_down=expect_SOblique,expect_SOblique
                mid_SOblique=expect_SOblique
                SOblique_bool=((av_SOblique>=mid_SOblique and mid_SOblique>=av_SOblique-sige_SOblique) or (av_SOblique<=mid_SOblique and mid_SOblique<=av_SOblique+sige_SOblique))

                expect_TOblique=T2HDMofAlphaBeta(i,k,k,alph,np.arctan(j),mW,mZ,higgs,Gf,alp_EM)
                #theoretical error very small here
                sig_TOblique=0
                expect_TOblique_up,expect_TOblique_down=expect_TOblique,expect_TOblique
                mid_TOblique=expect_TOblique
                TOblique_bool=((av_TOblique>=mid_TOblique and mid_TOblique>=av_TOblique-sige_TOblique) or (av_TOblique<=mid_TOblique and mid_TOblique<=av_TOblique+sige_TOblique))

                expect_UOblique=U2HDMofAlphaBeta(i,k,k,alph,np.arctan(j),mW,mZ,higgs,Gf,alp_EM,wangle)
                #theoretical error very small here
                sig_UOblique=0
                expect_UOblique_up,expect_UOblique_down=expect_UOblique,expect_UOblique
                mid_UOblique=expect_UOblique
                UOblique_bool=((av_UOblique>=mid_UOblique and mid_UOblique>=av_UOblique-sige_UOblique) or (av_UOblique<=mid_UOblique and mid_UOblique<=av_UOblique+sige_UOblique))

                if bs_bool and bd_bool and SOblique_bool and TOblique_bool and UOblique_bool:
                    i_log, j_log, k_log = np.log10(i),np.log10(j),np.log10(k)
                    mH_loc = np.append(mH_loc,i_log)
                    tanb_loc = np.append(tanb_loc,j_log)
                    mA_loc = np.append(mA_loc,k_log)
                    chit = chisq_simp([av_bsmu,av_bdmu,av_SOblique,av_TOblique,av_UOblique],[mid_bsmu,mid_bdmu,mid_SOblique,mid_TOblique,mid_UOblique],[sige_bsmu,sige_bdmu,sige_SOblique,sige_TOblique,sige_UOblique],[sig_bsmu,sig_bdmu,sig_SOblique,sig_TOblique,sig_UOblique])
                    chis = np.append(chis,chit)
                    if chit < chi_min[0]:
                        chi_min = [chit,i,j,k]

    return mH_loc, tanb_loc, mA_loc, chis, chi_min













