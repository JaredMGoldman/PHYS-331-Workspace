# Modules used

import numpy as np
import matplotlib.pyplot as plt

# Function definition section

# One-Dimensional Graphing Functions


def g(x):
	"""
	Newton-Raphson fixed-point function
	"""
	return x-(np.power(x,5)-3*np.power(x,3)+15*np.power(x,2)+27*x+9)/(5*np.power(x,4)-9*np.power(x,2)+30*x+27)


def dg(x):
	"""
	derivative
	"""
	return 1-(np.power((5*np.power(x,4)-9*np.power(x,2)+30*x+27),2)-(np.power(x,5)-3*np.power(x,3)+15*np.power(x,2)+27*x+9)*(20*np.power(x,3)-18*x+30))/(np.power((5*np.power(x,4)-9*np.power(x,2)+30*x+27),2))


def dg_abs(x):
	"""
	absolute value of the derivative
	"""
	return np.abs(1-(np.power((5*np.power(x,4)-9*np.power(x,2)+30*x+27),2)-(np.power(x,5)-3*np.power(x,3)+15*np.power(x,2)+27*x+9)*(20*np.power(x,3)-18*x+30))/(np.power((5*np.power(x,4)-9*np.power(x,2)+30*x+27),2)))


# Additional Functions to be Plotted


def diagonal(x):
    return x


def horizontal(x):				
    return 1+0*x


def HarryPlotter(f):
    """
    "The Chosen function"

    INPUT:
        f:   function, this is the function graphed
        xlo: float, lower bound of the graph of the function f
        xhi: float, upper bound of the graph of the function f
        func_num:   int, number of g function used
    OUTPUT:
        graph of function f over domain xlo-xhi with mesh size dx (definied within the function)
    """                                         
    xlo, xhi, ylo, yhi, dx = -10, 10, -10, 10, 1e-3                     # Plot generator function
    x_vals=np.arange(xlo,xhi+dx,dx)                                     # Determine x and y values
    y_vals= f(x_vals)
    plt.plot(x_vals,y_vals, label=f.__name__)
    plt.ylim(ylo,yhi)
    plt.grid()
    plt.legend(loc=0)
    plt.axhline(color='black')
    plt.axvline(color='black')
    plt.title("HW4 Extra Credit")
    plt.xlabel('x')
    plt.ylabel('y')


def plotcaller1(funcs):
	"""
	INPUT:
		funcs: list of functions, each function wished to be graph in increasing numerical order
	OUTPUTS:
		subplot of all functions from the funcs list
	"""
	for f in funcs:
		HarryPlotter(f)
	plt.show()


funcs =[ g, dg, dg_abs, diagonal, horizontal ]

plotcaller1(funcs)