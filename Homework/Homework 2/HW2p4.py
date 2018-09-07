import numpy as np 
from numpy import power as p
import matplotlib.pyplot as plt
from copy import deepcopy

def bracket(lo, hi):
    return ((lo+hi)/2)

def HarryPlotter(f,xlo,xhi,xtol):
    x_vals=np.arange(xlo,xhi,xtol)
    y_vals= f(x_vals)
    plt.plot(x_vals,y_vals)
    plt.grid()
    plt.title("Bernulli Equation")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def feelTheBern(h):
    Q=1.2
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
   
f, xlo, xhi, xtol, nmax = feelTheBern, 0.0, 0.4, 1e-6, 1e6

if type(rf_bisect(f, xlo, xhi, xtol, nmax)) == str:
    print(rf_bisect(f, xlo, xhi, xtol, nmax))
else:
    (root,iters) = rf_bisect(f, xlo, xhi, xtol, nmax)
    print('Root #1 of', f.__name__, ':', str(root))
    print('# iterations: ' + str(iters))
    fval=feelTheBern(root)
    print('feelTheBern evaluated at root is: ' + str(fval))
print("")


f, xlo, xhi, xtol, nmax = feelTheBern, 0.4, 0.8, 1e-6, 1e6

if type(rf_bisect(f, xlo, xhi, xtol, nmax)) == str:
    print(rf_bisect(f, xlo, xhi, xtol, nmax))
else:
    (root,iters) = rf_bisect(f, xlo, xhi, xtol, nmax)
    print('Root #2 of', f.__name__, ':', str(root))
    print('# iterations: ' + str(iters))
    fval=feelTheBern(root)
    print('feelTheBern evaluated at root is: ' + str(fval))