import numpy as np
import matplotlib.pyplot as plt


def g(x):
    return x-(np.power(x,5)-3*np.power(x,3)+15*np.power(x,2)+27*x+9)/(5*np.power(x,4)-9*np.power(x,2)+30*x+27)


def dg(x):
    return 1-(np.power((5*np.power(x,4)-9*np.power(x,2)+30*x+27),2)-(np.power(x,5)-3*np.power(x,3)+15*np.power(x,2)+27*x+9)*(20*np.power(x,3)-18*x+30))/(np.power((5*np.power(x,4)-9*np.power(x,2)+30*x+27),2))


def dg_abs(x):
    return np.abs(1-(np.power((5*np.power(x,4)-9*np.power(x,2)+30*x+27),2)-(np.power(x,5)-3*np.power(x,3)+15*np.power(x,2)+27*x+9)*(20*np.power(x,3)-18*x+30))/(np.power((5*np.power(x,4)-9*np.power(x,2)+30*x+27),2)))


def diagonal(x):
    return x


def horizontal(x):
    return 1+0*x


def HarryPlotter(f,func_num):
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
    xlo, xhi, ylo, yhi, dx = -5, 0, -3, 2, 1e-3                  #Plot generator function
    x_vals=np.arange(xlo,xhi+dx,dx)                #Determine x and y values
    y_vals= f(x_vals)
    plt.plot(x_vals,y_vals, label=f.__name__,linestyle='o')
    plt.ylim(ylo,yhi)
    plt.grid()
    plt.legend(loc=0)
    plt.axhline(color='black')
    plt.axvline(color='black')
    plt.title("Problem 1: Function "+ 'g'+str(func_num))
    plt.xlabel('x')
    plt.ylabel('y')


def funcshinup(funcs):
    i=0
    for packet in funcs:
        i+= 1
        func_num=i
        for f in packet:
            HarryPlotter(f,func_num)
        plt.show()


funcs =[ (g, dg, dg_abs, diagonal, horizontal) ]


funcshinup(funcs)