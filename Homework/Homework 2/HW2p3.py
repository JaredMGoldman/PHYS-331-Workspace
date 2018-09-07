"""
2. Modify rf_bisect to instead return a tuple of two numpy arrays containing the sequence 
of xmid and fmid (i.e., the sequence of middle points calculated by the algorithm and the 
corresponding values of the function, at each iteration).

3. For each of the functions f1(x), f2(x) and f3(x) of problem 2, compute the sequence of 
xmid and fmid values using a tolerance of 10-12. Make a plot of each (xmid, fmid) over the 
plot of the function itself. Use appropriate marker styles to visualize the individual data 
points of (xmid, fmid).

4. For each function, compute the error at each iteration by assuming the root at the final 
iteration is the “true” value. For the purpose of diagnostics, do not take the absolute value 
– just define error as the difference of the root at a given iteration from the “true” value. 
Display plots of this error versus the iteration number.
"""

import numpy as np
import matplotlib.pyplot as plt

def f2(x):
    return x**3

def f3(x):
    return np.sin(1. / (x + 0.01))

def f4(x):
    return 1. / (x - 0.5)

def bracket(lo, hi):
    return ((lo+hi)/2)

def PerryThePlotapus(f,x,xlo,xhi,xtol):
    x_array=np.asarray(x)
    y_array=f(x_array)
    x_vals=np.arange(xlo,xhi,1e-3)
    y_vals= f(x_vals)
    plt.scatter(x_array,y_array, color= "black")
    plt.plot(x_vals,y_vals)
    plt.grid()
    plt.title("Problem 3, Function: " + f.__name__)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return (x_array, y_array)


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
    x_array=[]
    iters=0
    low=deepcopy(xlo)
    high=deepcopy(xhi)
    try:
        while iters<= nmax:
            iters+=1
            if 0-f(bracket(low,high))<0: 
                x_array.append(bracket(low,high))
                high=deepcopy(bracket(low,high))
            elif 0-f(bracket(low,high))>0:
                x_array.append(bracket(low,high))
                low=deepcopy(bracket(low,high))
            if abs((0-f(bracket(low,high))))<= xtol:
                root=float(bracket(low,high))
                x_array.append(root)
                return PerryThePlotapus(f,x_array,xlo,xhi,xtol)
        return "Iteration limit reached, no root found."
    except ZeroDivisionError:
        print("Function", f.__name__, "has no root.")
for f in [f2,f3,f4]:
    print("For function " + f.__name__)
    rf_bisect(f,-1.,1., 1e-12, 1e5)
    print("")