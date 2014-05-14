import numpy as np
import pylab as plt


def hammer (x_obs, y_obs, guess, chi_2, step_size ,n_params, n_points):
    walk = np.empty((n_params,n_points))
    chi2 = np.empty((n_points))

    walk[:,0] = guess
    chi2[0] = chi_2(x_obs,y_obs,guess)
    
    for i in range(n_points - 1):
        params_init = walk[:,i]
        params_prime = np.empty((n_params))
        
        for j in range(n_params):
            params_prime[j] = np.random.normal(walk[j,i],step_size[j])

        chi_2_init = chi_2(x_obs, y_obs, params_init)
        chi_2_prime = chi_2(x_obs, y_obs, params_prime)
    
    
        alpha = -(chi_2_prime - chi_2_init)
            
        if (alpha>0.0):
            walk[:,i+1] = params_prime
            chi2[i+1] = chi_2_prime
        else:
            beta = np.random.random()
            if (alpha > np.log(beta)):
                walk[:,i+1] = params_prime
                chi2[i+1] = chi_2_prime
            else:
                walk[:,i+1] = params_init
                chi2[i+1] = chi_2_init
    
    
    best_index = np.argmin(chi2)
    best_value = walk[:,best_index]
    
    return best_value, walk, chi2