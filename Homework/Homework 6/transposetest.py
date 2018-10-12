import numpy as np

a = np.ndarray((1,5))

for i in range(0,5):
    num = 0
    a[i,0] = num
    num += 1

print(a)
print('')
print(np.transpose(a))