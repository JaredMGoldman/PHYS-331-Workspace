from scipy import interpolate as spi
import numpy as np
import matplotlib.pyplot as plt 

x_vals = np.array([1,2,3,4,5,6])
y_vals = np.array([2,4,2,4,2,4])
cubic = spi.interp1d(x_vals, y_vals, kind= 'cubic')
x_vals_fin =np.arange(1,6,0.001)
plt.plot(x_vals_fin, cubic(x_vals_fin),color='red')
plt.scatter(x_vals,y_vals, color='blue')
plt.show()