import numpy as np
import matplotlib.pyplot as plt

def h(t):
    return np.cos(np.pi*7*t)

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
    plt.title("Problem 1: Part a")
    plt.axhline(color='black')
    plt.axvline(color='black')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def main():
    xvals = np.linspace(0,30,300)
    hvals = h(xvals)
    Hn = np.fft.fft(hvals)
    print(np.size(Hn), type(Hn))
    plt.plot(xvals,np.abs(Hn))
    plt.grid()
    plt.title("Hn vals")
    plt.axhline(color='black')
    plt.axvline(color='black')
    plt.xlabel('t')
    plt.ylabel('H_n')
    plt.show()
    fn = np.linspace(0,10,300)
    plt.plot(fn,np.abs(Hn))
    plt.title("Hn vals")
    plt.axhline(color='black')
    plt.axvline(color='black')
    plt.xlabel('frequency')
    plt.ylabel('H_n')
    plt.grid()
    plt.show()


main()