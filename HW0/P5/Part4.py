import matplotlib.pyplot as plt
import numpy as np

x = np.array(range(1,100))
y = np.log2(x)
y2 = x-1

plt.plot(x, y, label="Infinite Cashiers")
plt.plot(x, y2, label="1 Cashier")
plt.legend(loc = 'upper left')
plt.xlabel("Number of bags")
plt.ylabel("Time needed (seconds)")

plt.show()

	
