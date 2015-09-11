import numpy as np
import matplotlib.pyplot as plt
import math

x = np.linspace(1, 150, num=50)
y1 = [int(math.ceil(math.log(i, 2))) for i in x]
y2 = [i - 1 for i in x]
y2[0] = 1

plt.plot(x, y1, 'bs')
plt.plot(x, y2, 'g^')
plt.xlabel('Number of Bags')
plt.ylabel('Counting Time (s)')
plt.title('P5 plot')
plt.show()
