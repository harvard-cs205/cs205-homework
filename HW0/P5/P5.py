import matplotlib.pyplot as plt
import numpy as np

# Compute the time with an infinite number of cashier
def inf_count(N):
	# Find the higher power in the binary representation
	rep = 1
	power = 0

	while N > rep:
		rep *= 2
		power += 1
	return power

# Ploting the result
N = 1000
t = np.arange(1, N, 1)

plt.plot(t, [inf_count(ti) for ti in t], '--r', label='Infinite number of Cashiers')
plt.plot(t, t -1, '--g', label='One cashier')
plt.legend()
plt.show()