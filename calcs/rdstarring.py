from __future__ import division 
import numpy as np
from fitting import *
from scipy.integrate import quad

g_gev = (1.1663787e-5)**2
hbar_gev = 6.582119514e-25
g_mev = (1.1663787e-11)**2
hbar_mev = 6.582118514e-22

def rdst(mBs,mD,rhod,r01,r11,r21,Vcb,mmu,mtau,vev,mc,mb,tanb,mH):
    '''
        R(D*)
    '''
    def rd1(s,mBs,mD,rhod,r01,r11,r21,Vcb,ml):
        hai=0.904
        PD = np.sqrt(s**2 + mBs**4 + mD**4 - 2*(s*mBs**2 + s*mD**2 + (mBs**2)*(mD**2)))/(2*mBs)
        ws = (mBs**2 + mD**2 - s)/(2*mBs*mD)
        weer = 2*np.sqrt(mBs*mD)/(mBs+mD)
        z = (np.sqrt(ws+1)-np.sqrt(2))/(np.sqrt(ws+1)+np.sqrt(2))
        ha1w = hai*(1 - 8*rhod*z + (53*rhod - 15)*z**2 - (231*rhod - 91)*z**3)
        r0w = r01 - 0.11*(ws-1) + 0.01*(ws-1)**2
        r1w = r11 - 0.12*(ws-1) + 0.05*(ws-1)**2
        r2w = r21 + 0.11*(ws-1) - 0.01*(ws-1)**2
        A0 = r0w*ha1w/weer
        A1 = (ws+1)*weer*ha1w/2
        A2 = r2w*ha1w/weer
        V = r1w*ha1w/weer
        Hpl = (mBs+mD)*A1 - (2*mBs*PD*V/(mBs+mD))
        Hmn = (mBs+mD)*A1 + (2*mBs*PD*V/(mBs+mD))
        H0 = (-1/(2*mD*np.sqrt(s)))*((4*pow(mBs*PD,2)/(mBs+mD))*A2 - (mBs**2 - mD**2 - s)*(mBs+mD)*A1)
        Hs = (2*mBs*PD*A0)/np.sqrt(s)
        #dgamsm = (g_gev/(96*(np.pi**3)*mBs**2))*((abs(Hpl)**2 + abs(Hmn)**2 + abs(H0)**2)*(1+(ml**2)/(2*s)) + (3*(ml**2)*(abs(Hs)**2)/(2*s)))*s*PD*(1-(ml**2)/s)**2 
        dgamsm = (g_gev*(Vcb**2)/(96*(np.pi**3)*mBs**2))*((abs(Hpl)**2 + abs(Hmn)**2 + abs(H0)**2)*(1+(ml**2)/(2*s)) + (3*(ml**2)*(abs(Hs)**2)/(2*s)))*s*PD*(1-(ml**2)/s)**2 
        return dgamsm
    def rd2(s,mBs,mD,rhod,ml,gcp,Vcb,mH,fls,flp,mb,mc):
        hai=0.904
        PD = np.sqrt(s**2 + mBs**4 + mD**4 - 2*(s*mBs**2 + s*mD**2 + (mBs**2)*(mD**2)))/(2*mBs)
        ws = (mBs**2 + mD**2 - s)/(2*mBs*mD)
        weer = 2*np.sqrt(mBs*mD)/(mBs+mD)
        z = (np.sqrt(ws+1)-np.sqrt(2))/(np.sqrt(ws+1)+np.sqrt(2))
        ha1w = hai*(1 - 8*rhod*z + (53*rhod - 15)*z**2 - (231*rhod - 91)*z**3)
        r0w = r01 - 0.11*(ws-1) + 0.01*(ws-1)**2
        A0 = r0w*ha1w/weer
        #dgammix = np.sqrt(g_gev/2)*(ml*gcp/(8*(np.pi**3)*mH**2))*((fls+flp)/(mb+mc))*(A0**2)*((1-(ml**2)/s)**2)*pow(PD,3)
        dgammix = np.sqrt(g_gev/2)*(ml*gcp*pow(Vcb,2)/(8*(np.pi**3)*mH**2))*((fls+flp)/(mb+mc))*(A0**2)*((1-(ml**2)/s)**2)*pow(PD,3)
        return dgammix
    def rd3(s,mBs,mD,rhod,ml,gcp,Vcb,mH,fls,flp,md,mc):
        hai=0.904
        PD = np.sqrt(s**2 + mBs**4 + mD**4 - 2*(s*mBs**2 + s*mD**2 + (mBs**2)*(mD**2)))/(2*mBs)
        weer = 2*np.sqrt(mBs*mD)/(mBs+mD)
        ws = (mBs**2 + mD**2 - s)/(2*mBs*mD)
        z = (np.sqrt(ws+1)-np.sqrt(2))/(np.sqrt(ws+1)+np.sqrt(2))
        ha1w = hai*(1 - 8*rhod*z + (53*rhod - 15)*z**2 - (231*rhod - 91)*z**3)
        r0w = r01 - 0.11*(ws-1) + 0.01*(ws-1)**2
        A0 = r0w*ha1w/weer
        #dgamh = (pow(gcp/(mH**2),2)/(16*np.pi**3))*((fls**2 + flp**2)/pow(mb+mc,2))*s*pow(A0,2)*pow(1-(ml**2)/s,2)*pow(PD,3)
        dgamh = (pow(gcp*Vcb/(mH**2),2)/(16*np.pi**3))*((fls**2 + flp**2)/pow(mb+mc,2))*s*pow(A0,2)*pow(1-(ml**2)/s,2)*pow(PD,3)
        return dgamh
    
    gcp = (-(mc/tanb) + mb*tanb)/(np.sqrt(2)*vev)
    fmus = -(mmu*tanb/(np.sqrt(2)*vev))
    fmup = fmus
    fts = -(mtau*tanb/(np.sqrt(2)*vev))
    ftp = fts
   
    top = mBs**2 + mD**2 - 2*mBs*mD

    mgamsm, err1 = quad(rd1,mmu**2,top,args=(mBs,mD,rhod,r01,r11,r21,Vcb,mmu))
    mgammix, err2 = quad(rd2,mmu**2,top,args=(mBs,mD,rhod,mmu,gcp,Vcb,mH,fmus,fmup,mb,mc))
    mgamh, err3 = quad(rd3,mmu**2,top,args=(mBs,mD,rhod,mmu,gcp,Vcb,mH,fmus,fmup,mb,mc))
    dmu = mgamsm+mgammix+mgamh
#    mV = (Vcb/mgamsm)/(1+(mgammix+mgamh)/mgamsm)

    tgamsm, err1 = quad(rd1,mtau**2,top,args=(mBs,mD,rhod,r01,r11,r21,Vcb,mtau))
    tgammix, err2 = quad(rd2,mtau**2,top,args=(mBs,mD,rhod,mtau,gcp,Vcb,mH,fts,ftp,mb,mc))
    tgamh, err3 = quad(rd3,mtau**2,top,args=(mBs,mD,rhod,mtau,gcp,Vcb,mH,fts,ftp,mb,mc))
    dtau = tgamsm+tgammix+tgamh
#    tV = (Vcb/tgamsm)/(1+(tgammix+tgamh)/tgamsm)

    return dtau/dmu
#    return tgamsm/mgamsm
#    return np.sqrt((mV+tV)/2)

def error_rds(mBs,mBs_err,mD,mD_err,rhod,rhod_err,r01,r01_err,r11,r11_err,r21,r21_err,Vcb,Vcb_err,mmu,mmu_err,mtau,mtau_err,vev,vev_err,mc,mc_err,mb,mb_err,tanb,mH):

    rds = rdst(mBs,mD,rhod,r01,r11,r21,Vcb,mmu,mtau,vev,mc,mb,tanb,mH)

    u1 = abs(rdst(mBs+mBs_err[0],mD,rhod,r01,r11,r21,Vcb,mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    l1 = abs(rdst(mBs+mBs_err[1],mD,rhod,r01,r11,r21,Vcb,mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    u2 = abs(rdst(mBs,mD+mD_err[0],rhod,r01,r11,r21,Vcb,mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    l2 = abs(rdst(mBs,mD+mD_err[1],rhod,r01,r11,r21,Vcb,mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    u3 = abs(rdst(mBs,mD,rhod+rhod_err[0],r01,r11,r21,Vcb,mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    l3 = abs(rdst(mBs,mD,rhod+rhod_err[1],r01,r11,r21,Vcb,mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    u4 = abs(rdst(mBs,mD,rhod,r01+r01_err[0],r11,r21,Vcb,mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    l4 = abs(rdst(mBs,mD,rhod,r01+r01_err[1],r11,r21,Vcb,mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    u5 = abs(rdst(mBs,mD,rhod,r01,r11+r11_err[0],r21,Vcb,mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    l5 = abs(rdst(mBs,mD,rhod,r01,r11+r11_err[1],r21,Vcb,mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    u6 = abs(rdst(mBs,mD,rhod,r01,r11,r21+r21_err[0],Vcb,mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    l6 = abs(rdst(mBs,mD,rhod,r01,r11,r21+r21_err[1],Vcb,mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    u7 = abs(rdst(mBs,mD,rhod,r01,r11,r21,Vcb+Vcb_err[0],mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    l7 = abs(rdst(mBs,mD,rhod,r01,r11,r21,Vcb+Vcb_err[1],mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    u8 = abs(rdst(mBs,mD,rhod,r01,r11,r21,Vcb,mmu+mmu_err[0],mtau,vev,mc,mb,tanb,mH)-rds)**2
    l8 = abs(rdst(mBs,mD,rhod,r01,r11,r21,Vcb,mmu+mmu_err[1],mtau,vev,mc,mb,tanb,mH)-rds)**2
    u9 = abs(rdst(mBs,mD,rhod,r01,r11,r21,Vcb,mmu,mtau+mtau_err[0],vev,mc,mb,tanb,mH)-rds)**2
    l9 = abs(rdst(mBs,mD,rhod,r01,r11,r21,Vcb,mmu,mtau+mtau_err[1],vev,mc,mb,tanb,mH)-rds)**2
    u10 = abs(rdst(mBs,mD,rhod,r01,r11,r21,Vcb,mmu,mtau,vev+vev_err[0],mc,mb,tanb,mH)-rds)**2
    l10 = abs(rdst(mBs,mD,rhod,r01,r11,r21,Vcb,mmu,mtau,vev+vev_err[1],mc,mb,tanb,mH)-rds)**2
    u11 = abs(rdst(mBs,mD,rhod,r01,r11,r21,Vcb,mmu,mtau,vev,mc+mc_err[0],mb,tanb,mH)-rds)**2
    l11 = abs(rdst(mBs,mD,rhod,r01,r11,r21,Vcb,mmu,mtau,vev,mc+mc_err[1],mb,tanb,mH)-rds)**2
    u12 = abs(rdst(mBs,mD,rhod,r01,r11,r21,Vcb,mmu,mtau,vev,mc,mb+mb_err[0],tanb,mH)-rds)**2
    l12 = abs(rdst(mBs,mD,rhod,r01,r11,r21,Vcb,mmu,mtau,vev,mc,mb+mb_err[1],tanb,mH)-rds)**2

    upper = np.sqrt(u1+u2+u3+u4+u5+u6+u7+u8+u9+u10+u11+u12)
    lower = np.sqrt(l1+l2+l3+l4+l5+l6+l7+l8+l9+l10+l11+l12)

    ups = rds+upper
    downs = rds-lower
    
    return ups, downs
    #return rds, [upper, -1*lower]

def itera_rds(mBs,mBs_err,mD,mD_err,rhod,rhod_err,r01,r01_err,r11,r11_err,r21,r21_err,Vcb,Vcb_err,mmu,mmu_err,mtau,mtau_err,vev,vev_err,mc,mc_err,mb,mb_err,rde,rde_err):

    sigma = 1
    rde_u,rde_d = rde+rde_err[0],rde+rde_err[1]
    av_rd = 0.5*(rde_u+rde_d)
    sige_rd = sigma*(rde_u-av_rd)
    log_mH_range = np.linspace(0,3.5,350)
    log_tanb_range = np.linspace(-1,2,300)
    mH_range = 10**log_mH_range
    tanb_range = 10**log_tanb_range
    mH_loc, tanb_loc, chi_rds = [],[],[]
    chi_rmin = 1000
    for i in mH_range:
        for j in tanb_range:
            #expect_branch = rdst(mBs,mD,rhod,r01,r11,r21,Vcb,mmu,mtau,vev,mc,mb,j,i)
            expect_branch_up, expect_branch_down = error_rds(mBs,mBs_err,mD,mD_err,rhod,rhod_err,r01,r01_err,r11,r11_err,r21,r21_err,Vcb,Vcb_err,mmu,mmu_err,mtau,mtau_err,vev,vev_err,mc,mc_err,mb,mb_err,j,i)
#            expect_branch_up, expect_branch_down = expect_branch+expect_error[0],expect_branch-expect_error[1]
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

def rdn(mBs,mD,rhod,delta,Vcb,mmu,mtau,vev,mc,mb,tanb,mH):
    '''
        R(D)
    '''
    def rdn1(s,mBs,mD,Vcb,ml,delta,rhod):
        PD = np.sqrt(s**2 + mBs**4 + mD**4 - 2*(s*mBs**2 + s*mD**2 + (mBs**2)*(mD**2)))/(2*mBs)
        V1i = 1.0611
        w = (mBs**2 + mD**2 - s)/(2*mBs*mD)
        z = (np.sqrt(w+1)-np.sqrt(2))/(np.sqrt(w+1)+np.sqrt(2))
        V1 = V1i*(1 - 8*rhod*z + (51*rhod - 10)*z**2 - (252*rhod - 84)*z**3)
        S1 = V1*(1 + delta*(-0.019 + 0.041*(w-1) - 0.015*(w-1)**2))
        F0 = (np.sqrt(mBs*mD)/(mBs+mD))*(w+1)*S1
        F1 = (np.sqrt(mBs*mD)*(mBs+mD)/(2*mBs*PD))*np.sqrt(w**2 - 1)*V1
        #dgamsm = (g_gev/(96*(np.pi**3)*mBs**2))*(4*pow(mBs*PD,2)*(1+(ml**2)/(2*s))*abs(F1)**2 + (3/(2*s))*(mBs**4)*pow((1-pow(mD/mBs,2))*ml*F0,2))*PD*(1-(ml**2)/s)**2
        dgamsm = (g_gev*(Vcb**2)/(96*(np.pi**3)*mBs**2))*(4*pow(mBs*PD,2)*(1+(ml**2)/(2*s))*abs(F1)**2 + (3/(2*s))*(mBs**4)*pow((1-pow(mD/mBs,2))*ml*F0,2))*PD*(1-(ml**2)/s)**2
        return dgamsm
    def rdn2(s,mBs,mD,rhod,delta,ml,gcs,Vcb,mH,fls,flp,mb,mc):
        PD = np.sqrt(s**2 + mBs**4 + mD**4 - 2*(s*mBs**2 + s*mD**2 + (mBs**2)*(mD**2)))/(2*mBs)
        w = (mBs**2 + mD**2 - s)/(2*mBs*mD)
        z = (np.sqrt(w+1)-np.sqrt(2))/(np.sqrt(w+1)+np.sqrt(2))
        V1i = 1.0611
        V1 = V1i*(1 - 8*rhod*z + (51*rhod - 10)*z**2 - (252*rhod - 84)*z**3)
        S1 = V1*(1 + delta*(-0.019 + 0.041*(w-1) - 0.015*(w-1)**2))
        F0 = (np.sqrt(mBs*mD)/(mBs+mD))*(w+1)*S1
        #dgammix = np.sqrt(g_gev/2)*(gcs/(32*np.pi**3))*pow(1/mH,2)*(fls+flp)*ml*(1-pow(mD/mBs,2))*((mBs**2 - mD**2)/(mb - mc))*(abs(F0)**2)*pow(1 - (ml**2)/s,2)*PD
        dgammix = np.sqrt(g_gev/2)*(gcs/(32*np.pi**3))*pow(Vcb/mH,2)*(fls+flp)*ml*(1-pow(mD/mBs,2))*((mBs**2 - mD**2)/(mb - mc))*(abs(F0)**2)*pow(1 - (ml**2)/s,2)*PD
        return dgammix
    def rdn3(s,mBs,mD,rhod,delta,ml,gcs,Vcb,mH,fls,flp,md,mc):
        PD = np.sqrt(s**2 + mBs**4 + mD**4 - 2*(s*mBs**2 + s*mD**2 + (mBs**2)*(mD**2)))/(2*mBs)
        w = (mBs**2 + mD**2 - s)/(2*mBs*mD)
        z = (np.sqrt(w+1)-np.sqrt(2))/(np.sqrt(w+1)+np.sqrt(2))
        V1i = 1.0611
        V1 = V1i*(1 - 8*rhod*z + (51*rhod - 10)*z**2 - (252*rhod - 84)*z**3)
        S1 = V1*(1 + delta*(-0.019 + 0.041*(w-1) - 0.015*(w-1)**2))
        F0 = (np.sqrt(mBs*mD)/(mBs+mD))*(w+1)*S1
        #dgamh = (pow(gcs/(mBs*mH**2),2)/(64*np.pi**3))*(fls**2 + flp**2)*(abs(F0)**2)*s*PD*(1 - (ml**2)/s)**2
        dgamh = (pow(gcs*Vcb/(mBs*mH**2),2)/(64*np.pi**3))*(fls**2 + flp**2)*(abs(F0)**2)*s*PD*(1 - (ml**2)/s)**2
        return dgamh
    
    gcs = ((mc/tanb) + mb*tanb)/(np.sqrt(2)*vev)
    fmus = -(mmu*tanb/(np.sqrt(2)*vev))
    fmup = fmus
    fts = -(mtau*tanb/(np.sqrt(2)*vev))
    ftp = fts
   
    top = mBs**2 + mD**2 - 2*mBs*mD

    mgamsm, err1 = quad(rdn1,mmu**2,top,args=(mBs,mD,Vcb,mmu,delta,rhod))
    mgammix, err2 = quad(rdn2,mmu**2,top,args=(mBs,mD,rhod,delta,mmu,gcs,Vcb,mH,fmus,fmup,mb,mc))
    mgamh, err3 = quad(rdn3,mmu**2,top,args=(mBs,mD,rhod,delta,mmu,gcs,Vcb,mH,fmus,fmup,mb,mc))
    dmu = mgamsm+mgammix+mgamh
#    mV = (Vcb/mgamsm)/(1+(mgammix+mgamh)/mgamsm)

    tgamsm, err1 = quad(rdn1,mtau**2,top,args=(mBs,mD,Vcb,mtau,delta,rhod))
    tgammix, err2 = quad(rdn2,mtau**2,top,args=(mBs,mD,rhod,delta,mtau,gcs,Vcb,mH,fts,ftp,mb,mc))
    tgamh, err3 = quad(rdn3,mtau**2,top,args=(mBs,mD,rhod,delta,mtau,gcs,Vcb,mH,fts,ftp,mb,mc))
    dtau = tgamsm+tgammix+tgamh
#    tV = (Vcb/tgamsm)/(1+(tgammix+tgamh)/tgamsm)

    return dtau/dmu
#    return tgamsm/mgamsm
#    return (tV+mV)/2

def error_rdn(mBs,mBs_err,mD,mD_err,rhod,rhod_err,delta,delta_err,Vcb,Vcb_err,mmu,mmu_err,mtau,mtau_err,vev,vev_err,mc,mc_err,mb,mb_err,tanb,mH):

    rds = rdn(mBs,mD,rhod,delta,Vcb,mmu,mtau,vev,mc,mb,tanb,mH)

    u1 = abs(rdn(mBs+mBs_err[0],mD,rhod,delta,Vcb,mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    l1 = abs(rdn(mBs+mBs_err[1],mD,rhod,delta,Vcb,mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    u2 = abs(rdn(mBs,mD+mD_err[0],rhod,delta,Vcb,mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    l2 = abs(rdn(mBs,mD+mD_err[1],rhod,delta,Vcb,mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    u3 = abs(rdn(mBs,mD,rhod+rhod_err[0],delta,Vcb,mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    l3 = abs(rdn(mBs,mD,rhod+rhod_err[1],delta,Vcb,mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    u6 = abs(rdn(mBs,mD,rhod,delta+delta_err[0],Vcb,mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    l6 = abs(rdn(mBs,mD,rhod,delta+delta_err[1],Vcb,mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    u7 = abs(rdn(mBs,mD,rhod,delta,Vcb+Vcb_err[0],mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    l7 = abs(rdn(mBs,mD,rhod,delta,Vcb+Vcb_err[1],mmu,mtau,vev,mc,mb,tanb,mH)-rds)**2
    u8 = abs(rdn(mBs,mD,rhod,delta,Vcb,mmu+mmu_err[0],mtau,vev,mc,mb,tanb,mH)-rds)**2
    l8 = abs(rdn(mBs,mD,rhod,delta,Vcb,mmu+mmu_err[1],mtau,vev,mc,mb,tanb,mH)-rds)**2
    u9 = abs(rdn(mBs,mD,rhod,delta,Vcb,mmu,mtau+mtau_err[0],vev,mc,mb,tanb,mH)-rds)**2
    l9 = abs(rdn(mBs,mD,rhod,delta,Vcb,mmu,mtau+mtau_err[1],vev,mc,mb,tanb,mH)-rds)**2
    u10 = abs(rdn(mBs,mD,rhod,delta,Vcb,mmu,mtau,vev+vev_err[0],mc,mb,tanb,mH)-rds)**2
    l10 = abs(rdn(mBs,mD,rhod,delta,Vcb,mmu,mtau,vev+vev_err[1],mc,mb,tanb,mH)-rds)**2
    u11 = abs(rdn(mBs,mD,rhod,delta,Vcb,mmu,mtau,vev,mc+mc_err[0],mb,tanb,mH)-rds)**2
    l11 = abs(rdn(mBs,mD,rhod,delta,Vcb,mmu,mtau,vev,mc+mc_err[1],mb,tanb,mH)-rds)**2
    u12 = abs(rdn(mBs,mD,rhod,delta,Vcb,mmu,mtau,vev,mc,mb+mb_err[0],tanb,mH)-rds)**2
    l12 = abs(rdn(mBs,mD,rhod,delta,Vcb,mmu,mtau,vev,mc,mb+mb_err[1],tanb,mH)-rds)**2

    upper = np.sqrt(u1+u2+u3+u6+u7+u8+u9+u10+u11+u12)
    lower = np.sqrt(l1+l2+l3+l6+l7+l8+l9+l10+l11+l12)

#    return rds, [upper, -1*lower]
    ups = rds+upper
    downs = rds-lower
    return ups, downs

def itera_rdn(mBs,mBs_err,mD,mD_err,rhod,rhod_err,delta,delta_err,Vcb,Vcb_err,mmu,mmu_err,mtau,mtau_err,vev,vev_err,mc,mc_err,mb,mb_err,rde,rde_err):

    sigma = 1
    rde_u,rde_d = rde+rde_err[0],rde+rde_err[1]
    av_rd = 0.5*(rde_u+rde_d)
    sige_rd = sigma*(rde_u-av_rd)
    log_mH_range = np.linspace(0,3.5,350)
    log_tanb_range = np.linspace(-1,2,300)
    mH_range = 10**log_mH_range
    tanb_range = 10**log_tanb_range
    mH_loc, tanb_loc, chi_rds = [],[],[]
    chi_rmin = 1000
    for i in mH_range:
        for j in tanb_range:
            #expect_branch = rdst(mBs,mD,rhod,r01,r11,r21,Vcb,mmu,mtau,vev,mc,mb,j,i)
            expect_branch_up, expect_branch_down = error_rdn(mBs,mBs_err,mD,mD_err,rhod,rhod_err,delta,delta_err,Vcb,Vcb_err,mmu,mmu_err,mtau,mtau_err,vev,vev_err,mc,mc_err,mb,mb_err,j,i)
#            expect_branch_up, expect_branch_down = expect_branch+expect_error[0],expect_branch-expect_error[1]
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

def itera_rda(mBs,mBs_err,mD,mD_err,mDs,mDs_err,rhod,rhod_err,r01,r01_err,r11,r11_err,r21,r21_err,delta,delta_err,Vcb,Vcb_err,mmu,mmu_err,mtau,mtau_err,vev,vev_err,mc,mc_err,mb,mb_err,rde,rde_err,rdste,rdste_err):

    sigma = 2
    rde_u,rde_d = rde+rde_err[0],rde+rde_err[1]
    rdste_u,rdste_d = rdste+rdste_err[0],rdste+rdste_err[1]
    av_rd = 0.5*(rde_u+rde_d)
    av_rdst = 0.5*(rdste_u+rdste_d)
    sige_rd = sigma*(rde_u-av_rd)
    sige_rdst = sigma*(rdste_u-av_rdst)
    log_mH_range = np.linspace(0,3.5,350)
    log_tanb_range = np.linspace(-1,2,300)
    mH_range = 10**log_mH_range
    tanb_range = 10**log_tanb_range
    mH_loc, tanb_loc, chi_rds = [],[],[]
    mHst_loc, tanbst_loc, chist_rds = [],[],[]
    chi_rmin,chist_rmin = 1000,1000
    for i in mH_range:
        for j in tanb_range:
            st_branch_up, st_branch_down = error_rds(mBs,mBs_err,mDs,mDs_err,rhod,rhod_err,r01,r01_err,r11,r11_err,r21,r21_err,Vcb,Vcb_err,mmu,mmu_err,mtau,mtau_err,vev,vev_err,mc,mc_err,mb,mb_err,j,i)
            expect_branch_up, expect_branch_down = error_rdn(mBs,mBs_err,mD,mD_err,rhod,rhod_err,delta,delta_err,Vcb,Vcb_err,mmu,mmu_err,mtau,mtau_err,vev,vev_err,mc,mc_err,mb,mb_err,j,i)
            mid_rd = 0.5*(expect_branch_up+expect_branch_down)
            mid_rdst = 0.5*(st_branch_up+st_branch_down)
            sig_rd = sigma*(expect_branch_up-mid_rd)
            sig_rdst = sigma*(st_branch_up-mid_rdst)
            rd_bool = ((av_rd >= mid_rd and mid_rd+sig_rd >= av_rd-sige_rd) or (av_rd <= mid_rd and mid_rd-sig_rd <= av_rd+sige_rd)) 
            rdst_bool = ((av_rdst >= mid_rdst and mid_rdst+sig_rdst >= av_rdst-sige_rdst) or (av_rdst <= mid_rdst and mid_rdst-sig_rdst <= av_rdst+sige_rdst)) 
            if rd_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mH_loc = np.append(mH_loc,i_log)
                tanb_loc = np.append(tanb_loc,j_log)
                chi_rij = chisq_simp([av_rd],[mid_rd],[sige_rd],[sig_rd])
                chi_rds = np.append(chi_rds,chi_rij)
                if chi_rij < chi_rmin:
                    chi_rmin = chi_rij

            if rdst_bool:
                i_log, j_log = np.log10(i), np.log10(j)
                mHst_loc = np.append(mHst_loc,i_log)
                tanbst_loc = np.append(tanbst_loc,j_log)
                chist_rij = chisq_simp([av_rdst],[mid_rdst],[sige_rdst],[sig_rdst])
                chist_rds = np.append(chist_rds,chist_rij)
                if chist_rij < chist_rmin:
                    chist_rmin = chist_rij

    return mH_loc, tanb_loc, chi_rds, chi_rmin, mHst_loc, tanbst_loc, chist_rds, chist_rmin
