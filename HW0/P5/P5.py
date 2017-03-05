import numpy as np
import matplotlib.pyplot as plt

plt.figure()

N = np.arange(0, 100)

plt.plot(N, N-1, '--b', label='lone cashier')
plt.plot(N, np.round_(np.log2(N)), '.g', label='infinite number of employees')

plt.xlabel('bag count')
plt.ylabel('counting time')

plt.show()