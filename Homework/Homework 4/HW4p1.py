import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.power(x,5)-3*np.power(x,3)+15*np.power(x,2)+27*x+9

def g1(x):
    return np.power(x,5)-3*np.power(x,3)+15*np.power(x,2)+28*x+9

def g2(x):
    return -np.power(x,5)+3*np.power(x,3)-15*np.power(x,2)-26*x-9

def g3(x):
    return (np.power(x,5)-3*np.power(x,3)+15*np.power(x,2)+25*x+9)/(-2)

def dg1(x):
    return 5*np.power(x,4)-9*np.power(x,2)+30*x+27

def dg2(x):
    return -5*np.power(x,4)+9*np.power(x,2)-30*x-26

def dg3(x):
    return (5*np.power(x,4)-9*np.power(x,2)-30*+25)/(-2)

def diagonal(x):
    return x

def horizontal(x):
    return 1+0*x

def dg1_abs(x):
    return np.abs(5*np.power(x,4)-9*np.power(x,2)+30*x+27)

def dg2_abs(x):
    return np.abs(-5*np.power(x,4)+9*np.power(x,2)-30*x-26)

def dg3_abs(x):
    return np.abs((5*np.power(x,4)-9*np.power(x,2)-30*+25)/(-2))

def HarryPlotter(f,xlo,xhi,func_num):
    """
    "The Chosen function"

    INPUT:
        f:   function, this is the function graphed
        xlo: float, lower bound of the graph of the function f
        xhi: float, upper bound of the graph of the function f
        func_num:   int, number of g function used   
    OUTPUT:
        graph of function f over domain xlo-xhi with mesh size dx (definied within the function)
    """
    dx=1e-3                                         #Plot generator function
    x_vals=np.arange(xlo,xhi+dx,dx)                 #Determine x and y values
    y_vals= f(x_vals)                               
    plt.plot(x_vals,y_vals, label=f.__name__)
    plt.grid()
    plt.legend(loc=0)
    plt.axhline(color='black')
    plt.axvline(color='black')
    plt.title("Problem 1: Function "+ 'g'+str(func_num))
    plt.ylim(xlo,xhi)
    plt.xlabel('x')
    plt.ylabel('y')

#----------------------------
#Part b


def funcshinup(funcs,xlo,xhi):
    i=0
    for packet in funcs:
        i+= 1
        func_num=i
        for f in packet:
            HarryPlotter(f,xlo,xhi,func_num)
        plt.show()

xlo, xhi, funcs = -10, 10, [(g1,dg1,dg1_abs,diagonal,horizontal),(g2,dg2,dg2_abs,diagonal,horizontal),(g3,dg3,dg3_abs,diagonal,horizontal)]

funcshinup(funcs, xlo, xhi)