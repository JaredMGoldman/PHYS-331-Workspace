import numpy as np
def calcSum_16bit(delta):
    sum = np.float16(0.0)
    delta = np.float16(delta)  # Convert delta to a 16-bit floating point number.

#    if delta == np.float16(0.0):  # If the 16-bit representation is exactly zero, throw an error.
#       print("Error: delta = " + str(delta) + " is equal to zero at this precision of 16 bits.")
#        return

    for i in range(0, int(round(1.0 / delta))):  # Compute the sum using a for loop.
        sum += delta
    return sum