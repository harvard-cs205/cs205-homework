import matplotlib.pyplot as plt
import numpy as np
import math

# Ploting the result
N = 1000
t = np.arange(1, N, 1)
epsilon = 10**(-6)  # Used to make the log discrete

plt.plot(t, [math.floor(math.log(ti, 2) - epsilon) + 1 for ti in t], '--r', label='Infinite number of Cashiers')
plt.plot(t, t -1, '--g', label='One cashier')
plt.legend()
plt.show()