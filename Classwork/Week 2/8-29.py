import numpy as np
import matplotlib.pyplot as plt
def graph_it():
    x=np.arange(-5,5,0.001)
    plt.plot(x,f(x))
    plt.ylim([10,10])
    plt.grid()
    plt.show()

def f(x):
    return(float(1+x**2-np.tan(x)))

print(graph_it())