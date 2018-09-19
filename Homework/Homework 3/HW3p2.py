import matplotlib.pyplot as plt
import numpy as np
from decimal import *

def f(x):
    return np.power(x,5)-3*np.power(x,3)+15*np.power(x,2)+27*x+9

def f_prime(x):
    return 5*np.power(x,4)-9*np.power(x,2)+30*x+27

def Newt(f,f_prime,x0,tol):
    i=0
    while abs(f(x0))>=tol:
        #print(str(x0),str(f(x0)),str(f_prime(x0)),str(i))
        i+=1
        x0 = Decimal(x0)-(Decimal(f(x0))/Decimal(f_prime(x0)))
        if i==30:
            return print("foo")
    print("One root of function:", f.__name__, "is at x =", str(x0), "Iteration Number:",str(i))
    

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
    plt.title("Problem 2")
    plt.axhline(color='black')
    plt.axvline(color='black')
    plt.xlabel('x')
    plt.ylim(-10,10)
    plt.ylabel('y')
    plt.show()
#for x in [0,-1,-2]:
 #   f, f_prime, x0, tol = f, f_prime, Decimal(x), 1e-15
  #  Newt(f,f_prime,x0,tol)
HarryPlotter(f,-4,1)