# HW7 Problem 2 template file -- Jared Goldman

#--Do not modify below this line--#
import numpy as np

# this function inputs an nxn matrix "Ainput" and an nx1 column matrix "binput"
# it performs Gaussian elimination and outputs the eliminated new matrices A and b
# where "A" is nxn upper triangular, and "b" is an nx1 column matrix
def GaussElimin(Ainput,binput):
    n=len(binput)
    A = Ainput.copy() # make copies so as not to write over originals
    b = binput.copy()
    # loop over pivot rows from row 1 to row n-1 (i to n-2)
    for i in range(0,n-1):
        # loop over row to be zero'ed from row j+1 to n (j+1 to n-1)
        for j in range(i+1,n):
            c = A[j,i]/A[i,i] # multiplicative factor to zero point
            A[j,i] = 0.0 # we know this element goes to zero
            A[j,i+1:n]=A[j,i+1:n]-c*A[i,i+1:n] # do subtraction of two rows
            b[j] = b[j]-c*b[i] # do substraction of b's as well
    return (A,b)  # return modified A and b

#---Do not modify above this line--#