import matplotlib.pyplot as plt
import numpy as np
import math as m

def f(x):
    return m.pow(x,3)-600
def f_prime(x):
    return m.pow(x,2)*3
def Newton_Raphson(x0):         #Buggy
    i=0
    while abs(m.pow(x0,3)-600)>=0.1:
        i+=1
        x0=((2*m.pow(x0,3)+600)/(3*m.pow(x0,2)))
        print(str(x0),":",str(f(x0)),"Iteration Number:",str(i))
def Newton_Raphson_Legit(f,f_prime,x0):
    i=0
    while abs(f(x0))>=0.1:
        i+=1
        x0=x0-f(x0)/f_prime(x0)
        print(str(x0),":",str(f(x0)),"Iteration Number:",str(i))
        print("")



x0=8
Newton_Raphson(x0)
#Newton_Raphson(f,f_prime,x0)
