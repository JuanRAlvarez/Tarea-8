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

def LotkaVolterra(state,t, alpha, beta, gamma, delta):
    x = state[0]
    y = state[1]
    xd = x*(alpha - beta*y)
    yd = -y*(gamma - delta*x)
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

guess = [10,10,10,10]
step_size = [0.1,0.1,0.1,0.1]

n_params = 4
n_points = 1000

best, walk, chi2 = MCMC.hammer(tiempos,[x_obs , y_obs] , guess, chi_2, step_size ,n_params, n_points)

print "El valor de alpha es", best[0]
print "El valor de beta es", best[1]
print "El valor de gamma es", best[2]
print "El valor de delta es", best[3]

f1=open('valores.dat', 'w+')

for i in range(n_points):
    f1.write('%f %f %f %f %f\n' %(walk[0,i], walk[1,i], walk[2,i], walk[3,i],chi2[i]))

plt.scatter(tiempos,y_obs,c='g')
plt.scatter(tiempos,x_obs,c='r')
plt.plot(t,my_model(t,best))
plt.xlabel('$Poblacion$')
plt.ylabel('$Tiempo$')
plt.title('Ajuste de los parametros')
plt.savefig("Ajuste.pdf")
plt.close()

