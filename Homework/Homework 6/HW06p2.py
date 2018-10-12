import numpy as np
import matplotlib.pyplot as plt


#-------------------------------
# Part a


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


#--------------------------------------
# Part b


def matrix_mult(m1,m2):
	"""
	INPUTS:
		m1,m2: numpy arrays of shape (m,n), (n,m) respectively where (m == n or m != n)
	OUTPUT:
		matrix product: numpy array of shape (m,n)

	Function that takes two matricies and returns their product.
	"""
	(r1,c1) = m1.shape
	(r2,c2) = m2.shape
	if c1 != r2:
		return "Error, incorect matricies computed. Please reenter matricies with appropriate dimensions."
	newentry=0
	matrix_product = np.zeros((c1,c2))
	for ra in range (0,r1):
		for cb in range (0, c2):
			for ca in range(0,c1):
				newentry = 0
				for rb in range(0,r2):
					newentry += (m1[ra,rb] * m2[rb,cb])
				matrix_product[ra,cb] = newentry			
	return matrix_product


def checkSolve(M, x, b):
    return matrix_mult(M,x)-b


#--------------------------------------
#part c


def mainc():
	M1 = np.array([[9,0,0],[-4,2,0],[1,0,5]])
	b1 = np.array([[8],[1],[4]])
	M2 = np.array([[2,4,5],[0,2,-4],[0,0,5]])
	b2 = np.array([[-4],[9],[4]])
	uL1 = 0
	uL2 = 1
	for vals in [(M1, b1, uL1), (M2,b2,uL2)]:
		(M,b,upperOrLower) = vals
		print('For matricies:')
		print('M =', M) 
		print("and")
		print('b =', b) 
		print('the residual is:')
		print(checkSolve(M,triSolve(M,b,upperOrLower),b))
		print('')


print('')
print('Part c')
mainc()
print('')
print('--------------------------')


#--------------------------------------
# Part d


def matrixgenerator():
	"""
	INPUTS:
		n: manual input, int, dimensions of triangular system
		upperOrLower: 1 or 0, determines wheter or not it is upper or lower triangular

	OUTPUTS:
		mymatrix: n x n array, upper or lower triangular (M)
		mymatrix1: n x 1 dimensional array, (b)

	"""
	n = int(input('Input a value, n, the dimension of the resulting matrix: '))
	upperOrLower = int(input('Input 0 for lower triangular and 1 for upper triangular matrix: '))
	
	mymatrix = np.zeros((n,n)) 
	mymatrix1 = np.zeros((n,1))

	if upperOrLower == 1:
		start = 0
		for i in range(0,n):
				first = True
				for j in range(start,n):
					if first:
						while abs(mymatrix[i,j]) <= 1e-5: 
							mymatrix[i,j] = np.random.randn()
					else:
						mymatrix[i,j] = np.random.randn()
					first = False
				start += 1

	if upperOrLower == 0:
		end = 1
		for i in range(0,n):
			for j in range(0,end):
				if j == (end - 1):
						while abs(mymatrix[i,j]) <= 1e-5: 
							mymatrix[i,j] = np.random.randn()
				else:
					mymatrix[i,j] = np.random.randn()
			end += 1
	else:
		return "Please reenter a value of 0 or 1 for the value of the triangular matrix"
	for i1 in range(0,n):
		mymatrix1[i1,0] = np.random.randn()

	return mymatrix, mymatrix1