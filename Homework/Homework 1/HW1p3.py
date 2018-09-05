#Jared Goldman

def maskn(lst, i):
    """
    inputs: 
        lst:    list of ints
        i:      single int
    output: 
        outlist: list of ints (0s and 1s)
    """
    outlist=[]                  #Defines the variable 'outlist' in order to build it
    for num in lst:             #Loop checks the terms of 'lst' for divisibility with i and adds the corresponding value to 'outlist.' This happens at 1:1 by iterating over 'lst' to ensure the values in 'outlist' are at the same index as their corresponding values in 'lst.'
        if num%i==0:
            outlist.append(1)
        else:
            outlist.append(0)
    return(outlist)