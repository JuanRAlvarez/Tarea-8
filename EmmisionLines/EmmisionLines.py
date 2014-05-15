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
guess = [5*(10**16), 1000, 1400, 10**2, -2]
step_size = [10**13, 10, 0.1, 0.1, 0.01]

n_params = 5
n_points = 500000

best, walk, chi2 = MCMC.hammer(x_obs, y_obs, guess, chi_2, step_size ,n_params, n_points)

print "El valor de A es", best[0]
print "El valor de B es", best[1]
print "El valor de E0 es", best[2]
print "El valor de sigma es", best[3]
print "El valor de alpha es", best[4]

plt.plot(walk[1,:],chi2)
plt.xlabel('$\\alpha$')
plt.ylabel('$\chi^2$')
plt.savefig("x2vsalpha.pdf")
plt.title('$\chi^2$ vs. $A$')
plt.close()
plt.plot(walk[0,:],walk[1,:])
plt.xlabel('$A$')
plt.ylabel('$B$')
plt.savefig("AvsB.pdf")
plt.close()
plt.plot(walk[0,:],walk[2,:])
plt.xlabel('$A$')
plt.ylabel('$E_0$')
plt.savefig("AvsE0.pdf")
plt.close()
plt.plot(walk[0,:],walk[3,:])
plt.xlabel('$A$')
plt.ylabel('$\sigma$')
plt.savefig("Avssigma.pdf")
plt.close()
plt.plot(walk[0,:],walk[4,:])
plt.xlabel('$A$')
plt.ylabel('$\\alpha$')
plt.savefig("Avsalpha.pdf")
plt.close()
plt.plot(walk[1,:],walk[2,:])
plt.xlabel('$B$')
plt.ylabel('$E_0$')
plt.savefig("BvsE0.pdf")
plt.close()
plt.plot(walk[1,:],walk[3,:])
plt.xlabel('$B$')
plt.ylabel('$\sigma$')
plt.savefig("Bvssigma.pdf")
plt.close()
plt.plot(walk[1,:],walk[4,:])
plt.xlabel('$B$')
plt.ylabel('$\\alpha$')
plt.savefig("Bvsalpha.pdf")
plt.close()
plt.plot(walk[2,:],walk[3,:])
plt.xlabel('$E_0$')
plt.ylabel('$\sigma$')
plt.savefig("E0vssigma.pdf")
plt.close()
plt.plot(walk[2,:],walk[4,:])
plt.xlabel('$E_0$')
plt.ylabel('$\\alpha$')
plt.savefig("E0vsalpha.pdf")
plt.close()
plt.plot(walk[3,:],walk[4,:])
plt.xlabel('$\sigma$')
plt.ylabel('$\\alpha$')
plt.savefig("sigmavsalpha.pdf")
plt.close()

plt.scatter(x_obs,y_obs)
plt.plot(x_obs,my_model(x_obs,best))
plt.xlabel('$N(E)$')
plt.ylabel('$E$')
plt.savefig("Ajuste.pdf")
plt.close()