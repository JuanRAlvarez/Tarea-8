import numpy as np
import pylab as plt
import fractions
from scipy.integrate import odeint
import MCMC


#IMPORTING!

datos = np.loadtxt("lotka_volterra_obs.dat")
x_obs = datos[:,1]
y_obs = datos[:,2]
tiempos = datos[:,0]

#TIMING!

def gcd(L):
    return reduce(fractions.gcd,L)

diferencias = [1000*tiempos[i]-1000*tiempos[i-1] for i in range(1,len(tiempos))]
paso = gcd(diferencias)/1000

t = np.arange(tiempos[0],tiempos[-1],paso)

#FINDING!

def find(element, LIST):
    return np.argmin(np.abs(LIST-element))

indices = [find(element, t) for element in tiempos]

#SOLVING

state0 = [datos[0,1],datos[0,2]]

def LotkaVolterra(state,t, alpha, beta, sigma, gamma):
    x = state[0]
    y = state[1]
    xd = x*(alpha - beta*y)
    yd = -y*(gamma - sigma*x)
    return [xd,yd]

#MODELLING

def my_model(t,params):
    alpha,beta, gamma, delta = params
    return odeint(LotkaVolterra,state0,t, args=(alpha,beta,gamma, delta))

def chi_2 (tiempos, X_obs, params):
    x_obs =X_obs[0]
    y_obs =X_obs[1]
    chi2_1 = sum((x_obs - my_model(tiempos,params)[:,0])**2)
    chi2_2 = sum((y_obs - my_model(tiempos,params)[:,1])**2)
    return chi2_1+chi2_2

guess = [20,5,5,40]
step_size = [0.01,0.01,0.01,0.01]

n_params = 4
n_points = 100000

best, walk, chi2 = MCMC.hammer(tiempos,[x_obs , y_obs] , guess, chi_2, step_size ,n_params, n_points)

print "El valor de alpha es", best[0]
print "El valor de beta es", best[1]
print "El valor de gamma es", best[2]
print "El valor de delta es", best[3]

plt.plot(walk[1,:],walk[0,:])
plt.xlabel('$\\alpha$')
plt.ylabel('$\\beta$')
plt.title('$\\alpha$ vs. $\\beta$')
plt.savefig("alphavsbeta.pdf")
plt.close()

plt.plot(walk[2,:],walk[0,:])
plt.xlabel('$\gamma$')
plt.ylabel('$\\alpha$')
plt.title('$\\alpha$ vs. $\gamma$')
plt.savefig("alphavsgamma.pdf")
plt.close()

plt.plot(walk[3,:],walk[0,:])
plt.xlabel('$\delta$')
plt.ylabel('$\\alpha$')
plt.title('$\\alpha$ vs. $\delta$')
plt.savefig("alphavsdelta.pdf")
plt.close()

plt.plot(walk[3,:],walk[2,:])
plt.xlabel('$\gamma$')
plt.ylabel('$\delta$')
plt.title('$\gamma$ vs. $\delta$')
plt.savefig("gammavsdelta.pdf")
plt.close()

plt.scatter(tiempos,y_obs)
plt.scatter(tiempos,x_obs)
plt.plot(t,my_model(t,best))
plt.xlabel('$Poblacion$')
plt.ylabel('$Tiempo$')
plt.title('Ajuste de los parametros')
plt.savefig("Ajuste.pdf")
plt.close()
