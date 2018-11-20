import numpy as np
import scipy.integrate as sci



def func4(x):
    return np.sqrt(1-x**2)/np.sqrt((np.sqrt(2)-1)**2-(np.sqrt(1+x**2)-1)**2)*(1/2)



def funcCheb(f,n):
    w = 0
    for i in range(n):
        x = np.cos(((2*i+1)*np.pi)/(2*(n-1)+2))         
        A= np.pi/n
        w += A*f(x)
    return w

print(funcCheb(func4,3))



def func5(x):
    return ((np.sqrt(2)-1)**2-(np.sqrt(1+x**2)-1)**2)**(-1/2)

print("\nActual Solution:", np.sqrt((sci.quad(func5,0,0.4)*2)))


def error(n):
    actual = sci.quad(func5,0,1)[0] 
    calculated = funcCheb(func4,n)
    if np.abs(calculated - actual) > 1e-6:
        print("n value",n," Error", np.abs(calculated - actual))
        return True
    else:
        return False

def mainb():
    n = 3
    while error(n):
        n += 1
        continue
    iterations = (n, funcCheb(func4,n))[0]
    print("The function converges after", iterations, "iterations.", np.abs(sci.quad(func5,0,1)[0]-funcCheb(func4,iterations)) )

mainb()





    
