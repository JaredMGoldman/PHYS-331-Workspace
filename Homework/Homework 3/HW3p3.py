# Template File for Homework 3, Problem 3
# PHYS 331
# Amy Oldenburg
#----------------------------------------------   
# DO NOT MODIFY BETWEEN THIS LINE AND ONE BELOW
## module newtonRaphson
import numpy as np
import matplotlib.pyplot as plt 

''' root = newtonRaphson(f,df,a,b,tol,maxiter).
    Finds a root of f(x) = 0 by combining the Newton-Raphson
    method with bisection. The root must be bracketed in (a,b).
    Calls user-supplied functions f(x) and its derivative df(x).   
    tol is the tolerance, and maxiter is the maximum number of iterations..
''' 
def newtonRaphson(f,df,a,b,tol,maxiter):  
    fa = f(a)
    if fa == 0.0: return a
    fb = f(b)
    if fb == 0.0: return b
    if np.sign(fa) == np.sign(fb): 
        print('Root is not bracketed')
        return []
    x = 0.5*(a + b)                    
    for i in range(maxiter):
        fx = f(x)
        if fx == 0.0: return x
      # Tighten the brackets on the root 
        if np.sign(fa) != np.sign(fx): b = x  
        else: a = x
      # Try a Newton-Raphson step    
        dfx = df(x)
      # If division by zero, push x out of bounds
        try: dx = -fx/dfx
        except ZeroDivisionError: dx = b - a
        x = x + dx
      # If the result is outside the brackets, use bisection  
        if (b - x)*(x - a) < 0.0:  
            dx = 0.5*(b - a)                      
            x = a + dx
      # Check for convergence     
        if abs(dx) < tol*max(abs(b),1.0): return x
    print('Too many iterations in Newton-Raphson')
## ADD your code below this line
#----------------------------------------------

def f1(x):
	"""
	Function for the natural frequencies 
	"""
	return np.cosh(x)*np.cos(x)+1

def f_prime(x):
	"""
	derivative of above function
	"""
	return np.cos(x)*np.sinh(x)-np.sin(x)*np.cosh(x)

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
	plt.title("Problem 3")
	plt.axhline(color='black')
	plt.axvline(color='black')
	plt.plot(x_vals, freq(x_vals))
	plt.ylim(-100,100)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()

def freq(x):                                        #Called when reporting the actual frequencies at the roots.
	L= 0.9
	E= 2e11
	I= 3.2552e-11
	m= 0.4415625
	return np.power((np.power(x,4)*E*I/(m*np.power(L,3))),0.5)/(2*np.pi)

a_b_vals={0.0:2.5,2.5:5.0}                  		#Dictionary of bounds of the brackets around each root.        
i=0
for a in a_b_vals.keys():                           #Calls the function around each root
	b= a_b_vals[a]
	i+=1
	f, df, tol, maxiter = f1, f_prime, 1e-8, 30
	print("")
	print('Fundemental frequency', str(i), 'is', str(freq(newtonRaphson(f,f_prime,a,b,tol,maxiter))), 'Hz', "beta:",str(newtonRaphson(f,f_prime,a,b,tol,maxiter)) )
HarryPlotter(f,-10,10)