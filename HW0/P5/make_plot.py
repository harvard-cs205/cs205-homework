from math import log, ceil
import pylab as plt

max_n = 300
x = range(1, max_n)
log_counting = lambda n: ceil(log(n, 2))
linear_counting = lambda n: n-1

plt.plot(x, map(log_counting, x), color='r', label='Infinite People')
plt.plot(x, map(linear_counting, x), color='steelblue', label='One person')
plt.legend(loc=2)
plt.xlabel('Number of Bags')
plt.ylabel('Time (seconds)')
plt.title('Time to count vs. number of people counting')


plt.show()