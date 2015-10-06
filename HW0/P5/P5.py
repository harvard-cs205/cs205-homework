import numpy as np
import matplotlib.pyplot as plt

N = np.linspace(0, 500, 500)
y = []

for n in N:
	y.append(np.log2(n))

plt.plot(N, N, label='Single cashier')
plt.plot(N, y, label='Infinite cashiers')
plt.legend()
plt.xlim(xmin=0)
plt.ylim(ymin=1)
plt.yscale('log')
plt.title('Performance comparison between infinite cashiers and single cashier')
plt.xlabel('# of bags')
plt.ylabel('Time to sum (seconds)')
plt.show()