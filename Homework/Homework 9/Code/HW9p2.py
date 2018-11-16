import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt


def f1(x):
    return np.sqrt(1-np.power(x,2))


def f2(x):
    return 0.5*np.power(x,-1.5)*np.sqrt(np.power(x,-0.5)*(2-np.power(x,-0.5)))


def func2():
    a, b = 0, 1
    print("Original integral is:", sci.quad(f1,a,b))

def func3():
    a,b = 1, 0
    print("Modified Integral is:", sci.quad(f2,a,b))

def mainc():
    func2()
    func3()
    dx = 1e-3
    x_vals = np.arange(0,1+dx,dx)
    plt.plot(x_vals, f1(x_vals), label = "Original Function")
    plt.plot(f2(x_vals), x_vals, label = "Modified Function")
    plt.grid()
    plt.xlabel("x")
    plt.xlim((0,1))
    plt.ylabel("y")
    plt.title("Problem 2, Part c: Integral Substitution")
    plt.legend()
    plt.axvline(color="black")
    plt.axhline(color="black")
    plt.show()

mainc()

