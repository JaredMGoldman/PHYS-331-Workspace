import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


rcParams['font.family']='Skia'


#--------------------
#Part b
 
def ratio(det,A):
	row, clmn = np.shape(A)
	norm = 0
	for i in range(row):
		for j in range(clmn):
			norm = np.power(A[i,j],2)
	norm = np.sqrt(norm)
	a = np.abs(det)
	return a/norm


def mymatrix(tvec):
	t1 = tvec[0]
	t2 = tvec[1]
	t3 = tvec[2]
	return np.array([[1,t1,t1**2/2],[1,t2,t2**2/2],[1,t3,t3**2/2]])


def checkConditioning(tvec):
	"""
	INPUT: 
		numpy array of shape (1,3), values of three different times to be conditioned

	OUTPUT:
		conditioning of function
	"""
	A = mymatrix(tvec)
	det = np.linalg.det(A)
	return ratio(det,A)


#------------------------
#Part c

def mainc():
	dx = 1e-2
	x_vals = np.array([])
	y_vals = np.array([])
	for tval in np.arange(0+dx,10,dx):
		tvec = np.array([0,tval, 10])
		ratio = checkConditioning(tvec)
		x_vals = np.append(x_vals, tval)
		y_vals = np.append(y_vals, ratio)
	plt.plot(x_vals ,y_vals, label = "Ratio as a Function of t_2")
	plt.legend()
	plt.title('HW7p1')
	plt.xlabel("t_2")
	plt.ylabel("Ratio")
	plt.axhline(color="black")
	plt.axvline(color="black")
	plt.grid()
	plt.show()

mainc()


#------------------------
# Part d


def matrixsolver(tvec, xvec):
	a = mymatrix(tvec)
	b = xvec
	params = np.linalg.solve(a,b)
	print('For matrix:', a, '. The best fit parameters for (x0, v0, a) are:', params)

def maind():
	i = 1
	scenarios = (((0, 1, 10), (0.3, 0.665, 14.3)), ((0, 5, 10), (0.3, 4.425, 14.3)))
	for vecs in scenarios:
		tvec, xvec = vecs
		print('xvec', xvec)
		print('tvec', tvec)
		print('Scenario', str(i) + ':')
		i += 1
		matrixsolver(tvec, xvec)

maind()


#-----------------------------
# Part e


def error(xvec):
	dx = 0.005
	xvec0 = xvec
	xvec_upper = np.array([xvec0[0], xvec0[1] + dx, xvec0[2] + dx])
	xvec_lower = np.array([xvec0[0], xvec0[1] - dx, xvec0[2] - dx])
	return xvec_lower, xvec_upper


def maine():
	i = 1
	scenarios = (((0, 1, 10), (0.3, 0.665, 14.3)), ((0, 5, 10), (0.3, 4.425, 14.3)))
	for vecs in scenarios:
		tvec, xvec = vecs
		print('Scenario', str(i) + ':')
		i += 1
		lower, upper = error(xvec)
		print('Lower Approximation:')
		matrixsolver(tvec, lower)
		print('')
		print('Upper Approximation')
		matrixsolver(tvec, upper)


maine()