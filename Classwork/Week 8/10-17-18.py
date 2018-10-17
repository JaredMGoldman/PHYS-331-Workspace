import numpy as np

A = np.random.randn(3,3)
(lam,P) = np.linalg.eig(A)

lam1 = lam[0]
w1 = P[:,0]

resid = np.dot(A,w1)-liam1*w1