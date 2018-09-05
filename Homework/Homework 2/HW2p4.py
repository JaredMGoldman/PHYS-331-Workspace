import numpy as np
import matplotlib.pyplot as plt

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

def feelTheBern(x):
    Q=1.2
    g=9.81
    b=1.8
    h0=0.6
    H=0.075
    return Q**2/(2*g*b**2*x**2)-Q**2/(2*g*b**2*h0**2)-h0+x+H
    

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
    from copy import deepcopy
    iters=0
    low=deepcopy(xlo)
    high=deepcopy(xhi)
    HarryPlotter(f,xlo,xhi,xtol)
    while iters<= nmax:
        iters+=1
        if 0-f(bracket(low,high))<0: 
            high=deepcopy(bracket(low,high))
        elif 0-f(bracket(low,high))>0:
            low=deepcopy(bracket(low,high))
        if abs((0-f(bracket(low,high))))<= xtol:
            root=float(bracket(low,high))
            print (root, iters)
            return (root, iters)
    return "Iteration limit reached, no root found."

(root,iters) = rf_bisect(feelTheBern, 0.0, 1., 1e-6, 1e5)
print('Root of feelTheBern: ' + str(root))
print('# iterations: ' + str(iters))
fval=feelTheBern(root)
print('feelTheBern evaluated at root is: ' + str(fval))