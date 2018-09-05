# Template File for Homework 2, Problem 1
# PHYS 331
# Amy Oldenburg
#-------------------------------------
import numpy as np
def calcSum_16bit(delta):
    sum = np.float16(0.0)
    delta = np.float16(delta)  # Convert delta to a 16-bit floating point number.

    if delta == np.float16(0.0):  # If the 16-bit representation is exactly zero, throw an error.
        print("Error: delta = " + str(delta) + " is equal to zero at this precision of 16 bits.")
        return

    for i in range(0, int(round(1.0 / delta))):  # Compute the sum using a for loop.
        sum += delta
    return sum

def calcSum_32bit(delta):
    sum = np.float32(0.0)
    delta = np.float32(delta)  # Convert delta to a 32-bit floating point number.

    if delta == np.float32(0.0):  # If the 32-bit representation is exactly zero, throw an error.
        print("Error: delta = " + str(delta) + " is equal to zero at this precision of 32 bits.")
        return

    for i in range(0, int(round(1.0 / delta))):  # Compute the sum using a for loop.
        sum += delta
    return sum

def calcSum_64bit(delta):
    sum = np.float64(0.0)
    delta = np.float64(delta)  # Convert delta to a 64-bit floating point number.

    if delta == np.float64(0.0):  # If the 64-bit representation is exactly zero, throw an error.
        print("Error: delta = " + str(delta) + " is equal to zero at this precision of 64 bits.")
        return

    for i in range(0, int(round(1.0 / delta))):  # Compute the sum using a for loop.
        sum += delta
    return sum
#--------------------------------

# Part b

for i in range(-1,-13,-1):
     print("For delta value of the float16-based function: " + "10^" + str(i) + " the sum of deltas that should equal one is: " + str(calcSum_16bit(10**i)))
print("")
for i in range(-1,-13,-1):
     print("For delta value of the float32-based function: " + "10^" + str(i) + " the sum of deltas that should equal one is: " + str(calcSum_32bit(10**i)))
print("")
for i in range(-1,-13,-1):
     print("For delta value of the float64-based function: " + "10^" + str(i) + " the sum of deltas that should equal one is: " + str(calcSum_64bit(10**i)))