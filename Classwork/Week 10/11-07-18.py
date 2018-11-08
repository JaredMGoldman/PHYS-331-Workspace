import scipy.integrate as scipy
import numpy as np

def func(x):
    return np.cos(x)**p

for p in range(1,6):
    print("Quadrature for p:", p, 2* np.cos(1/np.sqrt(3)) ** p)
    print("Integrate for p:", p, scipy.quad(func,-1,1))