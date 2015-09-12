import matplotlib.pyplot as plt
import math
X = range(2, 100)
# ceiling of log_2
Y = map(lambda x : int(math.log(x,2)) + 1, X)
# plot parallelized
plt.plot(X,Y)
# plot linear
plt.plot(X,X)
plt.title("Linear vs Parallelized counting")
plt.xlabel("Amount of bags to count; Green: Linear counting, Blue: Paralled counting")
plt.ylabel("Time in seconds")
plt.show()