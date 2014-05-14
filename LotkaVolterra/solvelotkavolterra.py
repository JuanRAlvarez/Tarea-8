from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


alpha=1
beta=4
gamma=5
delta=3
def system(z,t):
    x,y=z[0],z[1]
    dxdt= x*(alpha-beta*y)
    dydt=-y*(gamma-delta*x)
    return [dxdt,dydt]

t=np.linspace(0,30.,1000)
x0,y0=1.0,1.0
sol=odeint(system,[x0,y0],t)
X,Y=sol[:,0],sol[:,1]
plt.plot(t,X)
plt.plot(t,Y)
plt.show()