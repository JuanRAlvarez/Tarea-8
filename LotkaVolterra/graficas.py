import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

walk = np.loadtxt("valores.dat")

plt.scatter(walk[1,:],walk[0,:])
plt.xlabel('$\\alpha$')
plt.ylabel('$\\beta$')
plt.title('$\\alpha$ vs. $\\beta$')
plt.savefig("alphavsbeta.pdf")
plt.close()

plt.scatter(walk[2,:],walk[0,:])
plt.xlabel('$\gamma$')
plt.ylabel('$\\alpha$')
plt.title('$\\alpha$ vs. $\gamma$')
plt.savefig("alphavsgamma.pdf")
plt.close()

plt.scatter(walk[3,:],walk[0,:])
plt.xlabel('$\delta$')
plt.ylabel('$\\alpha$')
plt.title('$\\alpha$ vs. $\delta$')
plt.savefig("alphavsdelta.pdf")
plt.close()

plt.scatter(walk[3,:],walk[2,:])
plt.xlabel('$\gamma$')
plt.ylabel('$\delta$')
plt.title('$\gamma$ vs. $\delta$')
plt.savefig("gammavsdelta.pdf")
plt.close()

plt.scatter(np.log(walk[4,:]),walk[0,:])
plt.xlabel('$\chi^2$')
plt.ylabel('$\\alpha$')
plt.title('$\chi^2$ vs. $\\alpha$')
plt.savefig("x2vsalpha.pdf")
plt.close()

plt.scatter(np.log(walk[4,:]),walk[1,:])
plt.xlabel('$\chi^2$')
plt.ylabel('$\\beta$')
plt.title('$\chi^2$ vs. $\\beta$')
plt.savefig("x2vsbeta.pdf")
plt.close()

plt.scatter(np.log(walk[4,:]),walk[2,:])
plt.xlabel('$\chi^2$')
plt.ylabel('$\gamma$')
plt.title('$\chi^2$ vs. $\gamma$')
plt.savefig("x2vsgamma.pdf")
plt.close()

plt.scatter(np.log(walk[4,:]),walk[3,:])
plt.xlabel('$\chi^2$')
plt.ylabel('$\delta$')
plt.title('$\chi^2$ vs. $\delta$')
plt.savefig("x2vsdelta.pdf")
plt.close()

min_alpha = np.amin(walk[0,:])
max_alpha = np.amax(walk[0,:])
min_beta = np.amin(walk[1,:])
max_beta = np.amax(walk[1,:])
min_m = amin(m_walk)
max_m = amax(m_walk)
grid_m, grid_b = mgrid[min_m:max_m:200j, min_b:max_b:200j]

n_points = size(m_walk)
points = ones((n_points,2))
print shape(points)
points[:,0] = walk[0,:]
points[:,1] = walk[1,:]
grid_l = griddata(points, -log(walk[0,:]), (grid_m, grid_b), method='cubic')
plt.imshow(grid_l.T, extent=(min_m,max_m,min_b,max_b), aspect='auto',origin='lower')
plt.savefig("Densidadalphabeta.pdf")