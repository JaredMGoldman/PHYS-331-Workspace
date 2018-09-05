#Jared Goldman

from numpy import tanh, arange
import matplotlib.pyplot as plt

# Part a
def plot_tanh(xlim,dx):
    """
    inputs:
    xlim: float, half domain of function
    dx: float, step size of hyperbolic tangent function

    output:
    graph of hyperbolic tangent function through given x-vals
    """
    x_vals=arange(-xlim, xlim+dx, dx)               #Create an array of x values defined as 'x_vals'
    y_vals=(tanh(x_vals))                           #Create an array of y values defined as 'y_vals' based on 'x_vals'
    plt.plot(x_vals,y_vals)
    plt.legend(('tanh'),loc = 0)                    #Set up the graph
    plt.xlabel('x')
    plt.ylabel('TANH(x)')
    plt.grid()
    plt.title('HW1\np1')    
    plt.show()

# Part c
def plot_tanh2(a):
    """
    input:
        a: float, scalar
    output:
        plot
    """
    x_vals=arange(-5, 5.001, 0.001)                 #Create an array of x values defined as 'x_vals'
    y_vals=tanh((x_vals*a))                         #Create an array of y values defined as 'y_vals' based on 'x_vals'
    plt.plot(x_vals,y_vals) 
    plt.xlabel('x')                                 #Set up graph                       
    plt.ylabel('TANH(x*a)')
    plt.title('HW1\np1')
    plt.grid()

# Part b
print(plot_tanh(2,1))                               #Output specific functions      
print(plot_tanh(2,0.3))
print(plot_tanh(2,0.1))                             
print(plot_tanh(2,0.01))   

#Part d
plot_tanh2(0.5)                                     #Output specific functions
plot_tanh2(1.3) 
plot_tanh2(2.2)
plt.legend(("0.5","1.3","2.2"),loc = 0)
plt.show()