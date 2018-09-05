#Jared Goldman

from math import factorial

def taylor_sin(x0,n):
    """
    output:
        n-th order term in the taylor expansion series: float
    input:
        x0: float
        n: int
    """
    from math import factorial                              #Just in case
    if n<=0:                                                #Ensures that the number used for 'n' is within the permissible range
        return "Value Error, n must be greater than 0."
    if n%2==0:                                              #Returns 0 for the value of the taylor expansion term for all even orders of magnitude
        return 0
    elif (n+1)%4==0:                                        #Ensures the sign conventions are correct
        return(-(x0)**n/(factorial(n)))
    else:                                                   #Ensures the sign conventions are correct
        return ((x0)**n/(factorial(n)))