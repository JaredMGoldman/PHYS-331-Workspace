import numpy as np
import scipy.linalg as sci


def LUdecomp(Ainput):
    """
    this function inputs an nxn matrix "Ainput" and an nx1 column matrix "binput"
    it performs Gaussian elimination and outputs the eliminated new matrices A and b
    where "A" is nxn upper triangular, and "b" is an nx1 column matrix
    """

    n, m = np.shape(Ainput)
    
    if n != m:
        return 'Error: Please enter an invertible matrix.'
    
    U = Ainput.copy()                           # make copies so as not to write over originals
    L = np.zeros((np.shape(Ainput)))
    
    for i in range(0,n):
        L[i,i] = 1
    for i in range(0,n-1):                      # loop over pivot rows from row 1 to row n-1 (i to n-2)
        for j in range(i+1,n):                  # loop over row to be zero'ed from row j+1 to n (j+1 to n-1)
            c = U[j,i]/U[i,i]                   # multiplicative factor to zero point
            L[j,i] = c
            U[j,i] = 0.0                        # we know this element goes to zero
            U[j,i+1:n]=U[j,i+1:n]-c*U[i,i+1:n]  # do subtraction of two rows

    return (L,U)                                # return lower and upper decompositions


#--------------------
# Part b

def mainb():
    m0 = np.array([[4,-2,1],[-3,-1,4],[1,-1,3]], dtype = float)
    m1 = np.array([[4,-3,0,1],[2,2,3,2],[0,2,0,1],[6,1,-6,-5]], dtype = float)
    for A in [m0,m1]:
        L, U = LUdecomp(A)
        print('')
        print('Input:')
        print(A)
        print('')
        print('Calculated:')
        print(np.dot(L,U))
        print('')
        print('Lower:')
        print(L)
        print('')
        print("Upper:")
        print(U)
        print('')
        print('-------------------------')
        print('')


mainb()