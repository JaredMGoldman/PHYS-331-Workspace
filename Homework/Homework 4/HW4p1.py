# Modules Used
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# Proprietary function

def f(x):
    return np.power(x,5)-3*np.power(x,3)+15*np.power(x,2)+27*x+9


# Primary Functions 


def g1(x):
    return np.power(x,5)-3*np.power(x,3)+15*np.power(x,2)+28*x+9


def g2(x):
    return -np.power(x,5)+3*np.power(x,3)-15*np.power(x,2)-26*x-9


def g3(x):
    return (np.power(x,5)-3*np.power(x,3)+15*np.power(x,2)+25*x+9)/(-2)


# Derivitives


def dg1(x):
    return 5*np.power(x,4)-9*np.power(x,2)+30*x+27


def dg2(x):
    return -5*np.power(x,4)+9*np.power(x,2)-30*x-26


def dg3(x):
    return (5*np.power(x,4)-9*np.power(x,2)+30*x+25)/(-2)


# Absolute Value Functions of Derivatives


def dg1_abs(x):
    return np.abs(5*np.power(x,4)-9*np.power(x,2)+30*x+27)


def dg2_abs(x):
    return np.abs(-5*np.power(x,4)+9*np.power(x,2)-30*x-26)


def dg3_abs(x):
    return np.abs((5*np.power(x,4)-9*np.power(x,2)+30*x+25)/(-2))


# Additional Functions to be Plotted


def diagonal(x):
    return x


def horizontal(x):
    return 1+0*x


def HarryPlotter(f,xlo,xhi,func_num):
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
    dx=1e-3                                         #Plot generator function
    x_vals=np.arange(xlo,xhi+dx,dx)                 #Determine x and y values
    y_vals= f(x_vals)
    plt.plot(x_vals,y_vals, label=f.__name__)
    plt.grid()
    plt.legend(loc=0)
    plt.axhline(color='black')
    plt.axvline(color='black')
    plt.title("Problem 1: Function "+ 'g'+str(func_num))
    plt.ylim(xlo,xhi)
    plt.xlabel('x')
    plt.ylabel('y')

#----------------------------
# Part b


def plotcaller0(funcs,xlo,xhi):
	"""
	INPUT:
		funcs: list of tuples, each function wished to be graph in increasing numerical order
	OUTPUTS:
		subplot of all functions from the funcs list
	"""
	i=0
	for packet in funcs:
		i+= 1
		func_num=i
		for f in packet:
			HarryPlotter(f,xlo,xhi,func_num)
		plt.show()

xlo, xhi, funcs = -10, 10, [(g1,dg1,dg1_abs,diagonal,horizontal),(g2,dg2,dg2_abs,diagonal,horizontal),(g3,dg3,dg3_abs,diagonal,horizontal)]

plotcaller0(funcs, xlo, xhi)

#-----------------------------
# Part c


def fixed_pt(g,xstart,tol,nmax):
    try:
        x0=xstart                                               # Variable definition
        x= deepcopy(x0)
        i=0
        error_vals=np.array([])
        while i<=nmax:                                          # Set iteration limit
            i+=1
            x = g(x0)
            error_vals=np.append(error_vals, np.abs(x-x0))
            if abs(x-x0)<=tol:                                  # Check percision of x value
                error_v_iter(error_vals)
                return x
            x0= deepcopy(x)
        return "Iteration limit exceeded"                       # error plotting function not used in my code
    except ZeroDivisionError:
        print("An error has occured. Please try another function.")

def error_v_iter(error):
    """
    INPUT:
        error: numpy array, values of error per iteration
    OUTPUT:
        graph of error a function of iteration number
    """                                                         # Plot generator function
    x_vals=np.arange(1,len(error)+1)                          	# Determine x and y values
    y_vals=error
    plt.plot(x_vals,y_vals)
    plt.grid()
    plt.axhline(color='black')
    plt.axvline(color='black')
    plt.title("Error vs. Iteration")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.yscale('log')
    plt.show()

def CodeOfCalling(nmax, tol):
    """
    INPUT:
        nmax: int, number of max iterations for root finding functions below
        tol: float or int, tolerance of root finding functions
    OUTPUT:
        str: For function: < name > : < root or error message >
    """
    for g in [g1,g2,g3]:
        if g==g1:
            for xstart in [-0.77, -1.86]:
                print('For function',g.__name__,':',str(fixed_pt(g,xstart,tol,nmax)))
        if g==g2:
            for xstart in [-0.74,-1.87]:
                print('For function',g.__name__,':',str(fixed_pt(g,xstart,tol,nmax)))
        if g==g3:
            for xstart in [-0.69, -1.87]:
                print('For function',g.__name__,':',str(fixed_pt(g,xstart,tol,nmax)))

nmax, tol = 30, 1e-15

CodeOfCalling(nmax, tol)