import matplotlib.pyplot as plt
import numpy as np

x = np.array(range(1,300))
y = np.log2(x)
y2 = x-1

plt.plot(x, y, label="Infinite Employees")
plt.plot(x, y2, label="One Employee")
plt.legend(loc = 'upper left')
plt.xlabel("# of bags")
plt.ylabel("Time (sec)")

plt.show()