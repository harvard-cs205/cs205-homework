import matplotlib.pyplot as plt
import numpy as np
import math

def fx(x):
    return math.ceil(2*x**.5)-2

v_fx = np.vectorize(fx)

x = np.arange(1, 400)

plt.plot(x, v_fx(x), color="green", label="Best")
plt.plot(x, x-1, color="red", label="One Cashier")
plt.title("Bag Count and Counting Time")
plt.xlabel("Bag Count")
plt.ylabel("Counting Time")
plt.legend()
plt.show()
