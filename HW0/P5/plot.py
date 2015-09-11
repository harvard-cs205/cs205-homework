import matplotlib.pyplot as plt
import math
import numpy as np

x = np.linspace(1, 100, 100)
y1 = [i for i in x]
y2 = [math.log(i,2) for i in x]
plt.plot(x, y1)
plt.plot(x, y2)
plt.xlabel('Bag Count')
plt.ylabel('Time')
plt.show()