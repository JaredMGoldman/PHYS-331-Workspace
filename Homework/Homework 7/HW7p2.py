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

#----------------------------------
# Part b


def triSolve(M, b, upperOrLower):
    """
	INPUTS:
		M : numpy array, n x n upper or lower tiangular matrix
		b : numpy array, n x 1 dimensional
		upperOrLower: 1 or 0, corresponds to whether or not M is upper or Lower {1 : 'upper', 0 : 'lower'} 

	OUTPUT:
		xvals: numpy array, n x 1 dimensional, solution to triangular system
	"""
    n = np.shape(M)[0]
    xvals = np.zeros((n,1))
    n -= 1

    if upperOrLower == 1:
        for i in range (n,-1,-1):
            bval = b[i,0]
            aval = 0
            for j in range (0,n+1,1):
                aval += M[i,j]*xvals[j,0]
            cval = bval - aval
            try:
                xvals[i,0] = cval/M[i,i]
            except ZeroDivisionError:
                return 'Matrix M is not invertible.  Please submit an invertible matrix.'
                
    elif upperOrLower == 0:
        for i in range (0,n+1,1):
            bval = b[i,0]
            aval = 0
            for j in range (0,n+1,1):
                aval += M[i,j]*xvals[j,0]
            cval = bval - aval
            try:
                xvals[i,0] = cval/M[i,i]
            except ZeroDivisionError:
                return 'Matrix M is not invertible.  Please submit an invertible matrix.'

    return xvals


def mainb():
    A = np.array([[4,-2,1],[-3,-1,4],[1,-1,3]], dtype = float) 
    b = np.array([15,8,13], dtype = float)
    bnice = np.array([b])
    bnice = np.transpose(bnice)
    Atri, btri = GaussElimin(A,b)
    upperOrLower = 1
    btri = np.array([btri])
    btri = np.transpose(btri)
    xvec = triSolve(Atri, btri, upperOrLower)
    return xvec, A, bnice

def checkSolve(M, x, b):
    return matrix_mult(M,x)-b

x, M, b = mainb()

print(checkSolve(M,x,b))    # If this function returns an nx1 0 vector then the solution to the upper
                            # triangular is also a solution to the original system.
print("The x vector is:", x)


#---------------------
# Part c

def mainc():
    A = np.array([[4,-3,0,1],[2,2,3,2],[0,2,0,1],[6,1,-6,-5]], dtype = float)
    b = np.array([-7,-2,0,6], dtype = float)
    Atri, btri = GaussElimin(A,b)
    upperOrLower = 1
    btri = np.array([btri])
    btri = np.transpose(btri)
    xvec = triSolve(Atri, btri, upperOrLower)
    return xvec


print(mainc())