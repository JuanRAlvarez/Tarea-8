import numpy as np
import pylab as plt
import MCMC

def my_model (E, params):
    A, B, E0, sigma, alpha = params
    return A*(E**alpha) + B*np.exp(-((E-E0)/(np.sqrt(2)*sigma))**2)


def chi_2 (x_obs, y_obs, params):
    chi2 = sum((y_obs - my_model(x_obs,params))**2)
    return chi2


datos = np.loadtxt("energy_counts.dat")
x_obs = datos[:,0]
y_obs = datos[:,1]
guess = [10**16, 1000, 1400, 10**2, -4]
step_size = [10**13, 1, 1, 0.1, 10**-3]

n_params = 5
n_points = 100000

best, walk, chi2 = MCMC.hammer(x_obs, y_obs, guess, chi_2, step_size ,n_params, n_points)

print "El valor de A es", best[0]
print "El valor de B es", best[1]
print "El valor de E0 es", best[2]
print "El valor de sigma es", best[3]
print "El valor de alpha es", best[4]

plt.plot(walk[0,:],walk[1,:])
plt.show()
plt.plot(walk[2,:],walk[3,:])
plt.show()
plt.scatter(x_obs,y_obs)
plt.plot(x_obs,my_model(x_obs,best))
plt.show()