# #derived equations go to pdf

# import numpy as np
# import matplotlib.pyplot as plt
from copy import deepcopy
import scipy as sci
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np

def F(x_vec):
	"""
	INPUT:
		x_vec: numpy array, shape=(2,1)
	OUTPUT:
		numpy array shape = (2,1). Nonlinear transformation of x_vec over the function F(z)= 2(z)^3-3+i
	"""
	x = x_vec[0,0]
	y = x_vec[1,0]
	return np.array([[2*np.power(x,3)-6*x*np.power(y,2)-3],[6*np.power(x,2)*y-2*np.power(y,3)+1]])


def Jinv(x_vec):
	"""
	INPUT:
		x_vec: numpy array, shape = (2,1)
	OUTPUT:
		numpy array shape = (2,1)
	"""
	x = x_vec[0,0]
	y = x_vec[1,0]
	a = 6*np.power(x,2)-np.power(y,2)
	b = 12*x*y
	c = sci.power(6*np.power(x,2)+6*np.power(y,2),-2)
	return np.array([[a*c,b*c],[-1*b*c,a*c]])

def rf_newtonraphson2d(F_system, Jinv_system, x_vec0, tol, maxiter):
	for i in range(0,maxiter):
		dx = np.matmul(Jinv_system(x_vec0),F_system(x_vec0))
		x = F_system(x_vec0)[0,0]
		y = F_system(x_vec0)[1,0]
		if np.sqrt(np.power(x,2)+np.power(y,2)) <= tol:
			return x_vec0 
		else:
			x_vec0 = np.add(deepcopy(x_vec0), -1*dx)
	return "Error: Iteration limit exceeded."


# def rootfinder(r_vec,root):
# 	if root == 0:
# 		root_vec = np.array([[1.1582991206988713],[-0.12470628436608913]])
# 	elif root == 1:
# 		root_vec = np.array([[-0.4711515991533634],[1.0654703215184704]])
# 	elif root == 2:
# 		root_vec = np.array([[-0.6871487636032881],[-0.9407625743679481]])
# 	xr0 = r_vec[0,0]
# 	yr0 = r_vec[1,0]
# 	xr1 = root_vec[0,0]
# 	yr1 = root_vec[1,0]
# 	a = np.power(xr0-xr1,2)
# 	b = np.power(yr0-yr1, 2)
# 	return np.sqrt(a+b)


# my_array=np.array([[],[]])
# xlo, xhi, dx = -1.5, 1.5, 9e-3
# for x in np.arange(xlo,xhi+dx,dx):
# 	for y in np.arange(xlo,xhi+dx,dx):
# 		my_array=np.array([[x],[y]])
# 		F_system, Jinv_system, x_vec0, tol, maxiter = F, Jinv, my_array, 1e-5, 1000
# 		r_vec = rf_newtonraphson2d(F_system, Jinv_system, x_vec0, tol, maxiter)
# 		if rootfinder(r_vec,0) <= tol:
# 			plt.scatter(x,y,color='y')
# 		elif rootfinder(r_vec,1) <= tol:
# 			plt.scatter(x,y,color='m')
# 		elif rootfinder(r_vec,2) <= tol:
# 			plt.scatter(x,y,color='c')
# 		else:
# 			plt.scatter(x,y,c='k')
# plt.show()

# # my_array=np.array([[],[]])
# # xlo, xhi, dx = -1.5, 1.5, 8e-2
# # for x in np.arange(xlo,xhi+dx,dx):
# # 	for y in np.arange(xlo,xhi+dx,dx):
# # 		my_array=np.array([[x],[y]])
# # 		F_system, Jinv_system, x_vec0, tol, maxiter = F, Jinv, my_array, 1e-5, 1000
# # 		r_vec = rf_newtonraphson2d(F_system, Jinv_system, x_vec0, tol, maxiter)
# # 		xr = r_vec[0,0]
# # 		yr = r_vec[1,0]
# # 		mag = np.sqrt(np.power(xr,2)+np.power(yr,2))
# # 		plt.colormaps(x, y, mag, cmap = cm.viridis)
# # plt.show()




# make these smaller to increase the resolution
dx, dy = 0.05, 0.05

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[slice(-1.5, 1.5 + dy, dy),
                slice(-1.5, 1.5 + dx, dx)]

z = rf_newtonraphson2d(F, Jinv, np.array([[x],[y]]), 1e-5, 10000)

# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = z[:-1, :-1]
levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())


# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
cmap = plt.get_cmap('PiYG')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

fig, (ax0, ax1) = plt.subplots(nrows=2)

im = ax0.pcolormesh(x, y, z, cmap=cmap, norm=norm)
fig.colorbar(im, ax=ax0)
ax0.set_title('pcolormesh with levels')


# contours are *point* based plots, so convert our bound into point
# centers
cf = ax1.contourf(x[:-1, :-1] + dx/2.,
                  y[:-1, :-1] + dy/2., z, levels=levels,
                  cmap=cmap)
fig.colorbar(cf, ax=ax1)
ax1.set_title('contourf with levels')

# adjust spacing between subplots so `ax1` title and `ax0` tick labels
# don't overlap
fig.tight_layout()

plt.show()