import numpy as np
import matplotlib.pyplot as plt
# Get lots of values for N
N = np.array(range(1, 1000, 70))
# Calculate the logs
logs = np.ceil(np.log2(N))
# Plot the figure
plt.figure(1)
plt.scatter(N, logs, c='red', label='Infinite Employees')
plt.scatter(N, N-1, c='blue', label='One Employee')
plt.xlabel("Number of bags")
plt.ylabel("Time Needed (sec)")
plt.title("Comparison of Time Needed with Infinite Employees vs. One")
plt.legend()
plt.show()

