import numpy as np
import matplotlib.pyplot as plt


#-------------------------------
# Part a

def triSolve(M, b, upperOrLower):
    
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


print('Part a:')
print('')
print(triSolve(M,b,upperOrLower))
print('')
print('--------------------------')
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


print('Part b:')
print('')
print(checkSolve(M,triSolve(M,b,upperOrLower),b))
print('')
print('--------------------------')

#--------------------------------------
# Part d


def matrixgenerator(n, upperOrLower):
    
    # n = int(input('Input a value, n, the dimension of the resulting matrix: '))
    # upperOrLower = int(input('Input 0 for lower triangular and 1 for upper triangular matrix: '))
    
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
    
    for i1 in range(0,n):
        mymatrix1[i1,0] = np.random.randn()

    return mymatrix, mymatrix1

def main():
	for n in [3,10,30,100]:
		for upperOrLower in [1,0]:
			M, b = matrixgenerator(n,upperOrLower)
			print('')
			print('For a ' + str(n) + ' x ' + str(n) + ' matrix, the average error is: ' + str(np.sum(np.abs((checkSolve(M,triSolve(M,b,upperOrLower),b))))/np.size((checkSolve(M,triSolve(M,b,upperOrLower),b)))))
print('Part d:')   
main()
print('')
print('--------------------------')