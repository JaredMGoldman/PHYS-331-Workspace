import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.sqrt(np.cos(2*np.pi*x)+1)


def trap(f,a,b,n):
    h = (b-a)/(n-1)
    I = 0
    I += h/2*(f(a)+f(b))
    for n in np.arange(1,n-1):
        I += h*f(a+h*n)    
    return I


def simp13(f,a,b,n):
    I = 0
    while n % 6 != 1:
        n += 1
    print("Final value of n is:", n)
    nodd = np.arange(1,n,2)
    neven = np.arange(2,n-1,2)
    h = (b-a)/(n-1)
    I += f(a) + f(b)
    for n in nodd:
        I += 4*f(n*h+a)
    for n in neven:
        I += 2*f(n*h+a)
    I *= h/3
    return I

        

def simp38(f,a,b,n):
    I = 0
    while n % 6 != 1:
        n += 1
    print("Final value of n is:", n)
    h = (b-a)/(n-1)
    I += f(a) + f(b)
    n1 = np.arange(1,n-1,3)
    n2 = np.arange(2,n-1,3)
    n3 = np.arange(3,n-1,3)
    for n in n1:
        I += 3*f(n*h+a)
    for n in n2:
        I += 3*f(n*h+a)
    for n in n3:
        I += 2*f(n*h+a)
    I *= 3*h/8
    return I


def SimpsonIntegrate(f,a,b,n):
    """
    Input:
        f is the function to integrate. f should be defined such that it accepts a single floating point value and returns a single floating point value.
        a and b are the floating point limits of integration.
        n is the integer number of mesh points to perform the integration over.
    Output: 
        a tuple (I1,I2,I3), where I1 is the computed integral using the trapezoidal method, I2 is that using Simpson’s 1/3 rule, and I3 is that using Simpson’s 3/8 rule.
    """
    I1 = trap(f,a,b,n)
    I2 = simp13(f,a,b,n)
    I3 = simp38(f,a,b,n)
    return (I1,I2,I3)


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


#---------------------------
# Part a

def func1():
    a = 0
    b = 1
    HarryPlotter(f,a,b)
    return SimpsonIntegrate(f,a,b,10)

func1()


#---------------------------
# Part c

def mainc():
    a, b = 0, 1
    nvals = np.array([10,100,1000,3000,10000,30000,100000])
    trapvals = np.array([])
    simp13vals = np.array([])
    simp38vals = np.array([])
    x0 = 2*np.sqrt(2)/np.pi
    for n in nvals:
        (I1,I2,I3) = SimpsonIntegrate(f,a,b,n)
        trapvals = np.append(trapvals, np.abs(I1-x0))
        simp13vals = np.append(simp13vals, np.abs(I2-x0))
        simp38vals = np.append(simp38vals, np.abs(I3-x0))
    plt.plot(nvals, trapvals, label = "Trapezoidal")
    plt.plot(nvals, simp13vals, label = "Simpson's 1/3")
    plt.plot(nvals, simp38vals, label = "Simpson's 3/8")
    plt.title("Problem 1: Part c")
    plt.axvline(color = "black")
    plt.axhline(color = "black")
    plt.grid()
    plt.xlim((1e1,1e5))
    plt.legend()
    plt.loglog()
    plt.xlabel("Number of Mesh Points")
    plt.ylabel("Error")
    plt.show()

mainc()
