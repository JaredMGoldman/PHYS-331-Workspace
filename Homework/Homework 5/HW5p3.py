import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sci


def f(x):
	return np.abs(np.sin(x))


def HarryPlotter(f,xlo,xhi):
	"""
	"The Chosen function"

	INPUT:
		f: function, this is the function graphed
		xlo: float, lower bound of the graph of the function f
		xhi: float, upper bound of the graph of the function f

	OUTPUT:
		graph of function f over domain xlo-xhi with mesh size dx (definied within the function)
	"""
	dx=1e-1                                         #Plot generator function
	x_vals=np.arange(xlo,xhi+dx,dx)                 #Determine x and y values
	y_vals= f(x_vals)                               
	plt.scatter(x_vals,y_vals, c='violet')
	plt.grid()
	plt.title("Problem 3")
	plt.axhline(color='black')
	plt.axvline(color='black')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()


#--------------------
# Part b

HarryPlotter(f,-10,10)

#--------------------
# Part c


def mainc():
	xlo, xhi, dxm, dxs = -0.5, 0.5, 1e-3, 1e-1
	for meth in ('nearest','linear','quadratic','cubic'):
		x_master = np.arange(xlo,xhi + dxm, dxm)
		x_support = np.arange(xlo, xhi + dxs, dxs)
		f_interp = sci.interp1d(x_support, f(x_support), kind = meth, bounds_error = False)
		plt.plot(x_master, f(x_master), label='True Function', c = 'blue')
		plt.scatter(x_support, f(x_support), label='Support Points', c = 'green' )
		plt.plot(x_master, f_interp(x_master), label = 'Interpolation Function', c = 'purple')
		plt.grid()
		if meth == 'nearest':
			plt.title('Nearest Neighbor Interpolation')
		if meth == 'linear':
			plt.title('Linear Interpolation')
		if meth == 'quadratic':
			plt.title('Quadratic Interpolation')
		if meth == 'cubic':
			plt.title('Cubic Interpolation')
		plt.axhline(color='black')
		plt.axvline(color='black')
		plt.xlabel('x')
		plt.ylabel('y')
		plt.legend(loc=0)
		plt.show()


mainc()


#------------------------
# Part d

def maind():
	xlo, xhi, dxm, dxs = -np.pi, np.pi, 1e-3, 1e-1
	for meth in ('linear', 'cubic'):
		if meth == 'linear':
			lab = 'Linear'
		if meth == 'cubic':
			lab = 'Cubic'
		x_master = np.arange(xlo,xhi + dxm, dxm)
		x_support = np.arange(xlo, xhi + dxs, dxs)
		f_interp = sci.interp1d(x_support, f(x_support), kind = meth, bounds_error = False)
		plt.plot(x_master, f_interp(x_master), label = (lab + ' Interpolation Function'))
	plt.ylim(0.975,1.005)
	plt.xlim(-1.80,-1.35)
	plt.grid()
	plt.title('Linear vs. Cubic Interpolation')
	plt.axhline(color='black')
	plt.axvline(color='black')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.plot(x_master, f(x_master), label='True Function', linestyle = ':')
	plt.scatter(x_support, f(x_support), label='Support Points', c = 'green' )	
	plt.legend(loc=0)
	plt.show()


maind()