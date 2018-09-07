#Required libraries
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
#Single variable function section
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
    return ((lo+hi)/2)                          #midpoint finder

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
    plt.axhline(color='black')
    plt.axvline(color='black')
    plt.title("Problem 2: Function "+ f.__name__)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

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
    """                              #Function to call each combination of function and tolerence required for the problem
    for f in func:                          #calls each function from list
        print("For function "+ f.__name__)
        print("")
        for xtol in tol:                    #calls each tolerence from list
            xlo, xhi, nmax = -1., 1., 50
            try:                                                    #contingincy against potential asymptotic errors
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

FuncShinUp([f1,f2, f3, f4],[1e-3, 1e-6, 1e-12])         #Call it all