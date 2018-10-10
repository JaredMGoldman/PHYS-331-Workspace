# Jared Goldman: 
#-------------------------------------------
# DO NOT MODIFY
# Exam code: afjl9q935jgwj1844ga
#-------------------------------------------    
# Put All Import statements in this section
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sci
from decimal import Decimal
#-------------------------------------------
# Problem 1
def problem1():
    for num in range(1,202):
        if num % 7 == 0:
            print(num)
#roblem1()

#-------------------------------------------
# Problem 2
def myfunc(x):
    a = np.sin(10*x)
    b = np.sqrt(np.abs(x))
    c = np.power(x,6)+2
    return a-b/c

#-------------------------------------------
# Problem 3
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
    plt.plot(x_vals,y_vals, label = 'f(x)')
    plt.grid()
    plt.title("Practicum")
    plt.axhline(color='black')
    plt.axvline(color='black')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def f2(x):
    return 8*np.power(x,3)-2*np.power(x,2)+3


#HarryPlotter(f2,-2,2)

#-------------------------------------------
# Problem 4
def compute_distance(a0,a1):
    x0 = a0[0]
    y0 = a0[1]
    z0 = a0[2]
    x1 = a1[0]
    y1 = a1[1]
    z1 = a1[2]
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    return np.sqrt(np.power(dx,2) + np.power(dy,2) + np.power(dz,2))


#-------------------------------------------
# Problem 5
def search141(array):           # Accurate to roughly 1.4375e-14
    i=0
    for num in array:
        if Decimal(num)-Decimal(141.0) > 0.0:
            print("The first element above 141 was found at index: " + str(i) + ".")
            return  
        i += 1     
    print("No elements above 141 found.")

search141([141, 142,143,145])