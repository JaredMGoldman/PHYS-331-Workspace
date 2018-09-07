import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def f1(x):
    return 3 * x + np.sin(2*x) - np.exp(x)

def f2(x):
    return x**3

def f3(x):
    return np.sin(1. / (x + 0.01))

def bracket(lo, hi):
    """
    INPUT:
        lo: float, lower bound of bracket
        hi: float, upper bound of bracket
    
    OUTPUT:
        float, midpoint of the braketed region
    """
    return ((lo+hi)/2)

def PerryThePlotapus(f,x,xlo,xhi):
    """
    INPUT:
        f: function, this is the function graphed
        x: list, list of all midpoint values used to estimate root of function f
        xlo: float, lower bound of the graph of the function f
        xhi: float, upper bound of the graph of the function f
        xtol: float, the largest error accepted in the root value of f
    
    OUTPUT:
        graph of function f over the domain xlo to xhi with a mesh sixe dx (defined within the function)
        subplot of the scatter of the values in the 'x' input and their cooresponding values on the function f
        (x_array, y_array): tuple, (array form of 'x', array of cooresponding values on the function f)     
    """
    dx=1e-3
    x_array=np.asarray(x)
    y_array=f(x_array)
    x_vals=np.arange(xlo,xhi+dx,dx)
    y_vals= f(x_vals)
    plt.scatter(x_array,y_array, color= "black")
    plt.plot(x_vals,y_vals, color = "orange")
    plt.grid()
    plt.title("Problem 3: Function " + f.__name__)
    plt.xlabel('x')
    plt.axhline(color='black')
    plt.axvline(color='black')
    plt.ylabel('y')
    plt.show()
    errorGopher(f, x_array)
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
                return PerryThePlotapus(f,x_array,xlo,xhi)
        return "Iteration limit reached, no root found."
    except ZeroDivisionError:
        print("Function", f.__name__, "has no root.")

def errorGopher(f,xvals):
    """
    INPUT:
        f: function, one-dimensional
        xvals: array, x values used in bisection root search
    
    OUTPUT:
        graph of error as a function of iteration number
    """
    x_array = xvals
    true_val = x_array[-1]
    errorList=[]
    iterList=[]
    iter=0
    for x in x_array:
        iter+=1
        errorList.append(true_val-x)
        iterList.append(iter)
    plt.plot(iterList,errorList, color="orange")
    plt.title("Error as a Function of Iteration: Function " + f.__name__)
    plt.xlabel('Iterations')
    plt.ylabel('Error Values')
    plt.axhline(color="black")
    plt.axvline(color="black")
    plt.grid()
    plt.show()

for f in [f1,f2,f3]:
    print("For function " + f.__name__)
    print("")
    rf_bisect(f,-1.,1., 1e-12, 1e5)