# 2D Interpolation function

import numpy as np
import matplotlib.pyplot as plt 
def u(y):
    return (y-3)/3

def t(x):
    return (x-2)/3

def f(x,y):
    a1 = -3*(1-t(x))*(1-u(y))
    a2 = 4*t(x) *(1-u(y))
    a3 = 10*t(x)*u(y)
    a4 = 8*(1-t(x))*u(y)
    return a1 + a2 + a3 + a4

xr = np.arange(2,5.1,0.1)
yr = np.arange(3,6.1,0.1)

n = np.size(xr)
m = np.size(yr)

my_array = np.zeros([m,n])
for i in range(0,n):
    for j in range(0,m):
        my_array[j,i] = f(xr[i],yr[m-j-1])
plt.imshow(my_array)
plt.colorbar()
plt.show()