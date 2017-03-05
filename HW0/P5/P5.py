import matplotlib.pyplot as plt
import numpy as np

max_sz = 50

cashier = [N-1 for N in range(2, max_sz)]
infinite = [np.ceil(np.log(N)) for N in range(2,max_sz)]

# Plot the results
a = plt.plot(range(2,max_sz), cashier, marker='o')
b = plt.plot(range(2, max_sz), infinite, marker='o')
plt.legend(['Single cashier', 'Infinite employees'], loc='upper left')
plt.xlabel('Bag count')
plt.ylabel('Counting time')
plt.title('Speedup of using infinite employees vs. a single cashier')
plt.show()