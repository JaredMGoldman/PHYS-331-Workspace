#Jared Goldman

import copy

# Part b
def fib_loop(n):
    """
    input: 
        n, int greater than 0
    output:
        int, nth fibonacci number
    """                                                 #Define my variables 
    import copy                                         #Just in case
    prev_num=0                                          #'fib_num'  = the current fibonacci number
    fib_num=0                                           #'prev_num' = the most recent 'fib_num'
    i=0                                                 #   'i'     = the number of iterations through the while loop
    while i < n:                                        #Ensures that the for loop goes through the proper number of iterations to find the nth 'fib_num'
        if fib_num ==0:                                 #Deals with the initial condition as 'fib_num' always = 0 at the beginning of a function call
            fib_num=1
            i+=1                                        #Keeping track of iterations to ensure the while loop stops at the appropriate time            
        else:
            fib_num += prev_num                         #Changes the value of 'fib_num' to the next number in the fibonacci series
            prev_num = copy.deepcopy(fib_num-prev_num)  #Makes the previous number a copy of the previous value of 'fib_num' without running into aliasing issues by essenially reversing the last operation using copies of the variables
            i+=1                                        #Keeping track of iterations to ensure the while loop stops at the appropriate time
    return(fib_num)

# Part c
def fib_recur(n):
    """
    input:
        n, int >= 0
    output:
        int, nth fibonacci number
    """
    if n==0:                #Set initial conditions so that when the function is calssed, there will be base values to go back to
        return 0
    elif n==1:              #Also prvides an input if n is 0 or 1
        return 1
    else:
        return fib_recur(n-1) + fib_recur(n-2)  #Recursive sequence: calls functions and sums them until n-1=1 and n-2=0. Then it plugs in the values for the functions which have a greater 'n' value and eventually sums them all once their values are known.