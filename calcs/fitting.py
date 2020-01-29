from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def exp_like(err,x,xt):
    like = 1
    for i in range(len(x)):
        like_i = (1/(np.sqrt(2*np.pi)*err[i]))*np.exp(-0.5*((x[i]-xt[i])/err[i])**2)
        like = like*like_i
    return like


