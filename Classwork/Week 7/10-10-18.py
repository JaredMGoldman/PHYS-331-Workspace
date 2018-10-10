from scipy import linalg as sci
import numpy as np

# myarray = np.array([[1,2,3],[4,5,6],[7,8,9]])

# (L, U) = sci.lu(myarray, True)

#print(np.dot(L,U))

# randomarray = np.random.rand(10,10)

# print(randomarray)

# (L, U) = sci.lu(randomarray, True)

for n in range (5000, 100001, 10):
    randomarray = np.random.rand(n,n)
    (L, U) = sci.lu(randomarray, permute_l = True)
    print(n)