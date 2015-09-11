import numpy as np
import matplotlib.pyplot as plt
# Get lots of values for N
N = np.array(range(2, 1000, 10))
# Calculate the logs
logs = np.ceil(np.log2(N))
# Plot the figure
plt.figure(1)
plt.scatter(N, logs, c='red', label='Infinite Employees')
plt.scatter(N, N-1, c='blue', label='One Employee')
plt.xlabel("Number of bags")
plt.ylabel("Time Needed (sec)")
plt.yscale('log')
plt.axis([0, 1000, 0, 1000])
plt.title("Comparison of Time Needed with Infinite Employees vs. One")
plt.legend(loc='lower right')
plt.show()

