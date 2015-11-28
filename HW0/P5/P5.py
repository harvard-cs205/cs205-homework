# Plot bag count on the horizontal and counting time on the vertical. On the same graph, plot the time it would take the lone cashier to do this by himself.

import numpy as np
import matplotlib.pyplot as plt
import math

x = np.linspace(1,100,100)
y1 = [math.log(i) for i in x]
y2 = [i-1 for i in x]
plt.plot(x, y1, '-b', label='Infinite workers')
plt.plot(x, y2, '--g', label='Lone worker')
plt.xlabel('Bag count')
plt.ylabel('Counting time')
plt.title('Plot of counting times for infinite and lone worker(s)')
plt.legend(loc='best')
plt.show()
