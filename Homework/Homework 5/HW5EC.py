import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import scipy as sci

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


def rootfinder(r_vec,root):
	if root == 0:
		root_vec = np.array([[1.1582991206988713],[-0.12470628436608913]])
	elif root == 1:
		root_vec = np.array([[-0.4711515991533634],[1.0654703215184704]])
	elif root == 2:
		root_vec = np.array([[-0.6871487636032881],[-0.9407625743679481]])
	xr0 = r_vec[0,0]
	yr0 = r_vec[1,0]
	xr1 = root_vec[0,0]
	yr1 = root_vec[1,0]
	a = np.power(xr0-xr1,2)
	b = np.power(yr0-yr1, 2)
	return np.sqrt(a+b)

def main():
	my_array=np.array([[],[]])
	xlo, xhi, dx = -1.5, 1.5, 8e-2
	for x in np.arange(xlo,xhi+dx,dx):
		for y in np.arange(xlo,xhi+dx,dx):
			my_array=np.array([[x],[y]])
			F_system, Jinv_system, x_vec0, tol, maxiter = F, Jinv, my_array, 1e-5, 1000
			r_vec = rf_newtonraphson2d(F_system, Jinv_system, x_vec0, tol, maxiter)
			if rootfinder(r_vec,0) <= tol:
				plt.scatter(x,y,color='y')
			elif rootfinder(r_vec,1) <= tol:
				plt.scatter(x,y,color='m')
			elif rootfinder(r_vec,2) <= tol:
				plt.scatter(x,y,color='c')
			else:
				plt.scatter(x,y,c='k')
	plt.show()

main()
