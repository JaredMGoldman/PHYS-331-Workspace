import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rcParams


rcParams['font.family']='Skia'


#----------------------------
# Part b


def elem(num, vec):
    return vec[num,0]


def CreateSystem(kvec,mvec):
    
    n = np.shape(mvec)[0]
    mymatrix = np.zeros((n,n))
    m = n-1
    
    mymatrix[0,1] = elem(1,kvec)/elem(0,mvec)
    mymatrix[0,0] = -((elem(0,kvec)+elem(1,kvec))/elem(0,mvec))
    mymatrix[m,m-1] = elem(m,kvec)/elem(m,mvec)
    mymatrix[m,m] = -((elem(m,kvec)+elem(n,kvec))/elem(m,mvec))

    for i in range(1,m):
        mymatrix[i,i-1] = elem(i,kvec)/elem(i,mvec)
        mymatrix[i,i] = -((elem(i,kvec)+elem(i+1,kvec))/elem(i,mvec))
        mymatrix[i,i+1] = elem(i+1,kvec)/elem(i,mvec)


    return mymatrix


#--------------------------------
# Part c


def mainc():
    kvec = np.array([[1],[1],[1]])
    mvec = np.array([[1],[1]])
    mymatrix = CreateSystem(kvec,mvec)
    eigvals, eigvecs = np.linalg.eig(mymatrix)
    print('Part c')
    print('')
    print('For matrix:')
    print(mymatrix)
    print('')
    print('')

    print('Eigenvalues:')
    for val in eigvals:
        print(val)
        print('')
    print('')

    print('Eigenvectors:')
    for i in range(np.shape(eigvecs)[1]):
        print(eigvecs[:,i])
        print('')
    print('')
    print('----------------------')
    print('')


mainc()


# ---------------------------------
# Part d


def maind():
    kvec = np.array([[1],[1],[1],[1]])
    mvec = np.array([[1],[1],[1]])
    mymatrix = CreateSystem(kvec,mvec)
    eigvals, eigvecs = np.linalg.eig(mymatrix)
    
    print('Part d')
    print('')
    print('For matrix:')
    print(mymatrix)
    print('')
    print('')

    print('Eigenvalues:')
    for val in eigvals:
        print(val)
        print('')
    print('')

    print('Eigenvectors:')
    for i in range(np.shape(eigvecs)[1]):
        print(eigvecs[:,i])
        print('')
    print('')
    print('----------------------')
    print('')

maind()


#----------------------------------
# Part e


def maine():
    i = 1
    for mult in [1.5,1.2]:
        n = 1000
        kvec = np.ndarray((n+1,1))
        mvec = np.ndarray((n,1))
        for m in range(0,n):
            if m % 2 == 0: 
                mvec[m,0] = 1
            else:
                mvec[m,0] = mult
        for k in range(0,n+1):
            kvec[k,0] = 1

        mymatrix = CreateSystem(kvec,mvec)
        eigvals = np.linalg.eig(mymatrix)[0]
    
        plt.hist(eigvals, bins = 40)
        plt.title('Part e Plot '+ str(i))
        plt.xlabel('Eigenvalues')
        plt.ylabel('Count')
        plt.show()
        i += 1


maine()