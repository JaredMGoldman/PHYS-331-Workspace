#Necessary libraries
import numpy as np 
from numpy import power as p
import matplotlib.pyplot as plt
from copy import deepcopy

def HarryPlotter(f,xlo,xhi,xtol):
    """
    "The Chosen function"

    INPUT:
        f: function, this is the function graphed
        xlo: float, lower bound of the graph of the function f
        xhi: float, upper bound of the graph of the function f
    
    OUTPUT:
        graph of function f over domain xlo-xhi with mesh size dx (definied within the function)
    """
    x_vals=np.arange(xlo,xhi,xtol)              #Plot generator function
    y_vals= f(x_vals)                           #Determine x and y values
    plt.plot(x_vals,y_vals)
    plt.grid()
    plt.title("Bernulli Equation")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def feelTheBern(h):
    """
    INPUT: 
        h: array, domain of Bernulli's equation
    
    OUTPUT:
        array, range corresponding to domain 'h' for Bernulli's equation 
    """
    Q=1.2       #Bernulli's equation solved to equal zero with no h value in the denominator
    g=9.81
    b=1.8
    h0=0.6
    H=0.075
    return p(Q,2)*(p(h,2)-p(h0,2))+2*p(h0,2)*p(h,2)*p(b,2)*g*(h0-h-H)   
    

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
    iters=0
    low=deepcopy(xlo)
    high=deepcopy(xhi)
    HarryPlotter(f,xlo,xhi,xtol)
    #Bisection begins
    #Altered to be able to find both roots
    #Reads relative sign between midpoint and bracket instead of sign of the function at midpoint
    while iters<= nmax:             
        iters+=1
        if abs(f(bracket(low,high))+ f(high)) == (abs(f(bracket(low,high)))+ abs(f(high))): 
            high=deepcopy(bracket(low,high))  
        elif abs(f(bracket(low,high))+ f(low)) == (abs(f(bracket(low,high)))+ abs(f(low))):
            low=deepcopy(bracket(low,high))
        if abs(f(bracket(low,high)))<= xtol:
            root=float(bracket(low,high))
            return (root, iters)
    return "Iteration limit reached, no root found."

def bracket(lo, hi):
    """
    INPUT:
        lo: float, lower bound of bracket
        hi: float, upper bound of bracket
    
    OUTPUT:
        float, midpoint of the braketed region
    """
    return ((lo+hi)/2)  #Midpoint finder

#Call root one

f, xlo, xhi, xtol, nmax = feelTheBern, 0.0, 0.4, 1e-6, 50      #correct bracketing for root one

if type(rf_bisect(f, xlo, xhi, xtol, nmax)) == str:
    print(rf_bisect(f, xlo, xhi, xtol, nmax))
else:
    (root,iters) = rf_bisect(f, xlo, xhi, xtol, nmax)
    print('Root #1 of', f.__name__, ':', str(root))
    print('# iterations: ' + str(iters))
    fval=feelTheBern(root)
    print('feelTheBern evaluated at root is: ' + str(fval))
print("")

#Call root two

f, xlo, xhi, xtol, nmax = feelTheBern, 0.4, 0.8, 1e-6, 50    #correct bracketing for root two

if type(rf_bisect(f, xlo, xhi, xtol, nmax)) == str:
    print(rf_bisect(f, xlo, xhi, xtol, nmax))
else:
    (root,iters) = rf_bisect(f, xlo, xhi, xtol, nmax)
    print('Root #2 of', f.__name__, ':', str(root))
    print('# iterations: ' + str(iters))
    fval=feelTheBern(root)
    print('feelTheBern evaluated at root is: ' + str(fval))