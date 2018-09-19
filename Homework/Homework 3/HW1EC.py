import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def bracket(lo, hi):
    """
    INPUT:
        lo: float, lower bound of bracket
        hi: float, upper bound of bracket
    
    OUTPUT:
        float, midpoint of the braketed region
    """
    return ((lo+hi)/2)                          #midpoint finder

def rf_bisect(f,xlo,xhi,xtol,nmax):
    """
    Computes the value of the root for function f bracketed in the domain [xlo, xhi]. 
    PARAMETERS:
        f     --  (function) The one-dimensional function to evaluate a root of.
        xlo   --  (float) The lower limit of the bracket.
        xhi   --  (float) The upper limit of the bracket.
        xtol  --  (float) The tolerance the calculated root should achieve.
        nmax  --  (int) The maximum number of iterations allowed.

    RETURNS: (tuple(float, int)) A root of f that meets the tolerance tol the number 
    of iteratons required to converge.
    """
    iters, low, high = 0, deepcopy(xlo), deepcopy(xhi)  #variable definition section
    HarryPlotter(f,xlo,xhi)                       #get the graph      
    while iters< nmax:                            #Bisection search begins
        iters+=1
        if f(bracket(low,high))>0:                #determine position of midpoint relative to root
            high=bracket(low,high)
        elif f(bracket(low,high))<0:
            low=bracket(low,high)
        if abs(f(bracket(low,high)))<= xtol:      #Determine if the midpoint is the root
            root=float(bracket(low,high))
            return (root, iters)  
    return None

def f1(x):
    return np.power(x,3)-600

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
	dx=1e-3                                         #Plot generator function
	x_vals=np.arange(xlo,xhi+dx,dx)                 #Determine x and y values
	y_vals= f(x_vals)                               
	plt.plot(x_vals,y_vals)
	plt.grid()
	plt.title("Problem 1: EC")
	plt.axhline(color='black')
	plt.axvline(color='black')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()

f, xlo, xhi, xtol, nmax = f1, 8, 9, 0.1, 30
(root, iters)=rf_bisect(f,xlo,xhi,xtol,nmax)
print('The root of the function', f.__name__, 'is at x =', str(root),'after', str(iters),'iterations.')