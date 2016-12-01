import numpy as np
import matplotlib.pyplot as plt
bag = np.array(list(xrange(300))) + 1
t_single = bag - 1
t_infinite = np.ceil([np.log2(i) for i in bag])
plt.plot(bag, t_single, '-k', label='Single Cashier')
plt.plot(bag, t_infinite, '-b', label='Infinite Cashiers')
plt.legend()
plt.show()