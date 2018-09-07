"""
the interval x = (-1,1), with accuracy
tolerances of 10−3, 10−6, and 10−12
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

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
    iters, low, high = 0, deepcopy(xlo), deepcopy(xhi)
    HarryPlotter(f,xlo,xhi)
    while iters< nmax:
        iters+=1
        if 0-f(bracket(low,high))<0: 
            high=bracket(low,high)
        elif 0-f(bracket(low,high))>0:
            low=bracket(low,high)
        if abs(f(bracket(low,high)))<= xtol:
            root=float(bracket(low,high))
            return (root, iters)  
    return None

def f1(x):
    return 3 * x + np.sin(2*x) - np.exp(x)

def f2(x):
    return x**3

def f3(x):
    return np.sin(1. / (x + 0.01))

def f4(x):
    return 1. / (x - 0.5)

def bracket(lo, hi):
    """
    INPUT:
        lo: float, lower bound of bracket
        hi: float, upper bound of bracket
    
    OUTPUT:
        float, midpoint of the braketed region
    """
    return ((lo+hi)/2)

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
    dx=1e-3
    x_vals=np.arange(xlo,xhi+dx,dx)
    y_vals= f(x_vals)
    plt.plot(x_vals,y_vals)
    plt.grid()
    plt.title("Problem 2, Function: "+ f.__name__)
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.show()

def FuncShinUp(func,tol):
    """
    INPUT: 
        func: list, functions to find the roots of
        tol: list, different tolerences to try functions at
    
    OUTPUT:
        Name of function 
        Number of iterations taken to find root
        x-value of root
        y-value of root
        graph of function
    """
    for f in func:
        print("For function: "+ f.__name__)
        print("")
        for xtol in tol:
            xlo, xhi, nmax = -1., 1., 1e9
            try:
                if rf_bisect(f, xlo, xhi, xtol, nmax) == None:
                    print ("Iteration limit exceeded")
                else:
                    (root, iters)=rf_bisect(f, xlo, xhi, xtol, nmax)
                    print('Root of '+ f.__name__ + ': ' + str(root))
                    print('# iterations: ' + str(iters))
                    fval=f(root)
                    print(f.__name__ +' evaluated at root is: ' + str(fval))
                    print("")
            except ZeroDivisionError:
                print("Function", f.__name__, "has no root.")
                print("")
        print("")
        print("")

FuncShinUp([f1,f2, f3, f4],[1e-3, 1e-6, 1e-12])