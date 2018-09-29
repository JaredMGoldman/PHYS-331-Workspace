import numpy as np
import matplotlib.pyplot as plt

x_vals = np.array([0,1,2,3,4,5])
y_vals = np.array([4,2,-1,-1.5,2,-5])
x_vals1 = np.arange(0,5.001,0.0001)
weights = np.array([1,1,1,3,3,1])
y_vals1 = np.polyval(np.polyfit(x_vals,y_vals, 3),x_vals1)
plt.scatter(x_vals, y_vals, label= 'Original')
plt.plot(x_vals1, y_vals1, label = 'Best Fit Function', linestyle = '--')
plt.legend()
plt.axvline()
plt.axhline()
plt.ylabel('y')
plt.xlabel('x')
#plt.title("Best Fit Test")


y_vals1 = np.polyval(np.polyfit(x_vals,y_vals, 3, w= weights),x_vals1)
plt.scatter(x_vals, y_vals, label= 'Original')
plt.plot(x_vals1, y_vals1, label = 'Best Fit Function', linestyle = '--')
plt.legend()
plt.axvline()
plt.axhline()
plt.ylabel('y')
plt.xlabel('x')
plt.title("Weighted Best Fit Test")
plt.show()