import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import scipy as sci

def F(x_vec):
	"""
	INPUT:
		x_vec: numpy array, shape=(1,0)
	OUTPUT:
		numpy array shape = (1,0). Nonlinear transformation of x_vec over the function F(z)= 2(z)^3-3+i
	"""
	x = x_vec[0,0]
	y = x_vec[1,0]
	return np.array([[2*np.power(x,3)-6*x*np.power(y,2)-3],[6*np.power(x,2)*y-2*np.power(y,3)+1]])


def Jinv(x_vec):
	"""
	INPUT:
		x_vec: numpy array, shape = (1,0)
	OUTPUT:
		numpy array shape = (1,1)
	"""
	x = x_vec[0,0]
	y = x_vec[1,0]
	a = 6*np.power(x,2)-np.power(y,2)
	b = 12*x*y
	c = sci.power(6*np.power(x,2)+6*np.power(y,2),-2)
	return np.array([[a*c,b*c],[-1*b*c,a*c]])

def rf_newtonraphson2d(F_system, Jinv_system, x_vec0, tol, maxiter):
	"""
	INPUTS:
		F_system:		Main transformation function, takes a 2d numpy array vector.
		Jinv_system:	Inverse of Jacobian of the output of F_system, taked a 2d numpy array vector.
		x_vec0:			2d numpy array vector of the form [[x],[y]].
		tol:			Float, desired accuracy of roots.
		maxiter:		int, iteration limit of root-finding function.
	
	OUTPUT:
		x_vec:			2d numpy array vector of root.
	
	Multi-dimensional root-finding function.
	"""
	x_vec = x_vec0
	for i in range(0,maxiter):
		dx = matrix_mult(Jinv_system(x_vec),F_system(x_vec))
		x = F_system(x_vec)[0,0]
		y = F_system(x_vec)[1,0]
		if np.sqrt(np.power(x,2)+np.power(y,2)) <= tol:
			return x_vec 
		else:
			x_vec = np.add(deepcopy(x_vec), -1*dx)
	return "Error: Iteration limit exceeded."


def matrix_mult(m1,m2):
	"""
	INPUTS:
		m1,m2: numpy arrays of shape (m,n), (n,m) respectively where (m == n or m != n)
	OUTPUT:
		matrix product: numpy array of shape (m,n)

	Function that takes two matricies and returns their product.
	"""
	(r1,c1) = m1.shape
	(r2,c2) = m2.shape
	if c1 != r2:
		return "Error, incorect matricies computed. Please reenter matricies with appropriate dimensions."
	newentry=0
	matrix_product = np.zeros((c1,c2))
	for ra in range (0,r1):
		for cb in range (0, c2):
			for ca in range(0,c1):
				newentry = 0
				for rb in range(0,r2):
					newentry += (m1[ra,rb] * m2[rb,cb])
				matrix_product[ra,cb] = newentry			
	return matrix_product


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

# def main():
# 	"""
# 	INPUTS:
# 		NONE
# 	OUTPUTS:
# 		Scatterplot of colors cooresponding to each root.
	
# 	Function caller.
# 	"""
# 	my_array = np.array([[],[]])
# 	z_vals = np.array([])
# 	x_vals = np.array([])
# 	y_vals = np.array([])
# 	xlo, xhi, dx = -1.5, 1.5, 8e-2
# 	for x in np.arange(xlo,xhi+dx,dx):
# 		for y in np.arange(xlo,xhi+dx,dx):
# 			my_array=np.array([[x],[y]])
# 			F_system, Jinv_system, x_vec0, tol, maxiter = F, Jinv, my_array, 1e-5, 1000
# 			np.append(z_vals, rf_newtonraphson2d(F_system, Jinv_system, x_vec0, tol, maxiter))
# 			np.append(x_vals, x)
# 			np.append(y_vals, y)
# 	plt.pcolormesh(np.meshgrid(x_vals), z_vals, cmap = 'viridis' )
# 	plt.show()

# main()

# xlo, xhi, dx = -1.5, 1.5, 1e-3
# x0, y0 = np.arange(xlo, xhi + dx, dx), np.arange(xlo, xhi + dx, dx)
# xv, yv = np.meshgrid(x0,y0)
# z =  np.frompyfunc(rf_newtonraphson2d(F, Jinv, xv, yv, 1e-5, 1000))
# h = plt.contourf(x,y,z)
# plt.show()

def f(x, y):
	z = complex(x,y)
	return np.power(z,3)-3 +complex(0,1)

def df(x, y):
	z = complex(x,y)
	return np.power(z, 2)*3


def nrcmpx(xlo, xhi, dx, func, dfunc, tol, nmax):
	x_vals = np.array([])
	y_vals = np.array([])
	z_vals = np.array([])
	for x in np.arange(xlo, xhi+dx, dx):
		for y in np.arange(xlo, xhi+dx, dx):
			z = func(x, y)
			for iter in range(0,nmax):
				dz = -1*func(x,y)/dfunc(x,y)
				if np.abs(func(x, y)) <= tol:
					np.append(x_vals, x)
					np.append(y_vals, y)
					np.append(z_vals, z)
					break
				z += dz
	plt.imshow(np.array(x_vals, y_vals, z_vals), cmap = 'viridis')
	plt.colorbar()
	plt.show()


xlo, xhi, dx, func, dfunc, tol, nmax = -1.5, 1.5, 1e-1, f, df, 1e-5, 1000
nrcmpx(xlo, xhi, dx, func, dfunc, tol, nmax)



