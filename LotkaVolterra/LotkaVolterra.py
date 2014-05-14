import numpy as np
import pylab as plt

def my_model (t, params):
    alpha, beta, gamma, delta,  = params
    return A*(E**alpha) + B*np.exp(-((E-E0)/(np.sqrt(2)*sigma))**2)


def chi_2 (x_obs, y_obs, params):
    chi2 = sum((y_obs - my_model(x_obs,params))**2)
    return chi2