import numpy as np
def calcSum_64bit(delta):
    sum = np.float64(0.0)
    delta = np.float64(delta)  # Convert delta to a 64-bit floating point number.

    if delta == np.float64(0.0):  # If the 64-bit representation is exactly zero, throw an error.
        print("Error: delta = " + str(delta) + " is equal to zero at this precision of 64 bits.")
        return

    for i in range(0, int(round(1.0 / delta))):  # Compute the sum using a for loop.
        sum += delta
    return sum

print(calcSum_64bit(10**-308))
