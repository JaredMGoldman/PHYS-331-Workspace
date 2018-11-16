import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt


def error(I):
    return np.abs(I-np.pi/4)


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
    nodd = np.arange(1,n-1,2)
    neven = np.arange(2,n-2,2)
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


def f1(x):
    return np.sqrt(1-np.power(x,2))


def f2(x):
    return 2*np.power(x,2)*np.sqrt(2-np.power(x,2))


def func2():
    a, b = 0, 1
    print("Original integral is:", sci.quad(f1,a,b))

def func3():
    a,b = 1, 0
    print("Modified Integral is:", sci.quad(f2,a,b))

def mainc():
    n = 1e3
    # func2()
    # func3()
    # dx = 1e-3
    # x_vals = np.arange(0,1+dx,dx)
    # plt.plot(x_vals, f1(x_vals), label = "Original Function")
    # plt.plot(f2(x_vals), x_vals, label = "Modified Function")
    # plt.grid()
    # plt.xlabel("x")
    # plt.xlim((0,1))
    # plt.ylabel("y")
    # plt.title("Problem 2, Part c: Integral Substitution")
    # plt.legend()
    # plt.axvline(color="black")
    # plt.axhline(color="black")
    # plt.show()
    func1out = SimpsonIntegrate(f1,0,1,n)
    func2out = SimpsonIntegrate(f2,0,1,n)
    print("\nFor Original function:\n\nTrapezoid:", error(func1out[0]), "\nSimpson's 1/3:", error(func1out[1]), "\nSimpson's 3/8:", error(func1out[2]))
    print("\n-----------------\n\nFor Modified function:\n\nTrapezoid:", error(func2out[0]), "\nSimpson's 1/3:", error(func2out[1]), "\nSimpson's 3/8:", error(func2out[2]))

mainc()

