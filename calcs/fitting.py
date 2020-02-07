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

def chi_del(chi_min,chis,hs,ts):
    '''
        computes delta chisq, for several CLs? so 2 sigma so 95.45 just now
    '''
    delt_chis = chis-chi_min
    h_min,t_min = [],[]
    for i in range(len(hs)):
        if delt_chis[i] <= 5.99:
            h_min = np.append(h_min,hs[i])
            t_min = np.append(t_min,ts[i])

    return h_min, t_min

    





