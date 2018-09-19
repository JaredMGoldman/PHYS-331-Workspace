# Template File for Homework 3, Problem 4
# PHYS 331
# Amy Oldenburg

## module newtonRaphson

# has been modified to strip bisection aspects of the code
# is generally UNSAFE, but can be used for specific case of Problem 4
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def newtonRaphsonMOD(f,df,a,b): # YES YOU MAY MODIFY!
    fa = f(a)
    if fa == 0.0: return a
    fb = f(b)
    if fb == 0.0: return b
    if np.sign(fa) == np.sign(fb): 
        print('Root is not bracketed')
        return []
    x = 0.5*(a + b)                    
    sigma_array=[]
    for i in range(30):
        fx = f(x)
      # Try a Newton-Raphson step    
        dfx = df(x)
      # If division by zero, push x out of bounds
        try: dx = -fx/dfx
        except ZeroDivisionError: dx = b - a
        xNew= deepcopy(x) + dx
        sigma_array.append(abs(x-xNew))
        x = deepcopy(xNew)
        print(sigma_array)
    x_iter_vals=np.arange(1,(len(sigma_array)+1))
    np.asarray(sigma_array)
    plt.plot(x_iter_vals,sigma_array)
    plt.axhline(color='black')
    plt.axvline(color='black')
    plt.grid()
    plt.xlim(0,8)
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Problem 4: Error as a Function of Iteration')
    plt.yscale('log')
    plt.show()
    return sigma_array

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
    y2_vals= f1_prime(x_vals)
    plt.plot(x_vals,y_vals)
    plt.plot(x_vals,y2_vals)
    plt.grid()
    plt.title("Problem 4")
    plt.axhline(color='black')
    plt.axvline(color='black')
    plt.xlabel('x')
    #plt.ylim(-10,10)
    plt.ylabel('y')
    plt.show()

def f1(x):
    return (x+10)*(x-25)*(np.power(x,2)+45)
def f1_prime(x):
    return 4*np.power(x,3)-45*np.power(x,2)-410*x-675

newtonRaphsonMOD(f1,f1_prime,0,40)
HarryPlotter(f1,0,40)