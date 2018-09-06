"""
the interval x = (-1,1), with accuracy
tolerances of 10−3, 10−6, and 10−12
"""

import numpy as np
import matplotlib.pyplot as plt

def bracket(lo, hi):
    return ((lo+hi)/2)

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
    HarryPlotter(f,xlo,xhi)
    low=deepcopy(xlo)
    high=deepcopy(xhi)
    while iters< nmax:
        iters+=1
        if 0-f(bracket(low,high))<0: 
            high=bracket(low,high)
        elif 0-f(bracket(low,high))>0:
            low=bracket(low,high)
        if abs(f(bracket(low,high)))<= xtol:
            root=float(bracket(low,high))
            return (root, iters)  
    HarryPlotter(f,xlo,xhi)
    return None

def f1(x):
    return 3 * x + np.sin(2*x) - np.exp(x)

def f2(x):
    return x**3

def f3(x):
    return np.sin(1. / (x + 0.01))

def f4(x):
    """
    i=0
    print("x_array is:")
    print(x)
    for num in x:
        if num == 0.5:
            np.remove(x[i])
        else:
            i+=1
    """        
    #print("x_val")
    #print(x)
    return 1. / (x - 0.5)

def HarryPlotter(f,xlo,xhi):
    x_vals=np.arange(xlo,xhi,1e-3)
    print("x shit:")
    print(x_vals)
    y_vals= f(x_vals)
    plt.plot(x_vals,y_vals)
    plt.grid()
    plt.title("Problem 2, Function: "+ f.__name__)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

for f in [f4]:
    print("For function: "+ f.__name__)
    for xtol in [1e-3, 1e-6, 1e-12]:
        if rf_bisect(f,-1.,1.,xtol,19) == None:
            print ("Iteration limit exceeded")
        else:
            (root, iters)=rf_bisect(f, -1., 1., xtol, 19)
            print('Root of '+ f.__name__ + ': ' + str(root))
            print('# iterations: ' + str(iters))
            #fval=f(root)
            #print(f.__name__ +' evaluated at root is: ' + str(fval))
    print("")