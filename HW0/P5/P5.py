import numpy as np
import matplotlib.pyplot as plt

getTime = np.vectorize(lambda x: np.log2(x))
Nmax = 3000
xArray = np.linspace(2, 2000, 40)
plt.plot(xArray, getTime(xArray), marker = ".", label = "Infinite employees")
plt.plot(xArray, xArray - 1, marker = ".", linestyle = "--", label = "One Employee")
plt.legend(loc = "best")
plt.xlabel("# Bags (N)")
plt.ylabel("Time (in Sec.)")
plt.show()
