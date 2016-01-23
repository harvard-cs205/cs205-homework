import matplotlib.pyplot as plt
import math

x = range(1, 512)
# calculate series
y_inf = [math.log(i, 2) for i in x]
y_alone = [i-1 for i in x]

plt.plot(x, y_inf, '-b', label='Infinite Employees')
plt.plot(x, y_alone, '--g', label='Lone Cashier')
plt.yscale('log')
plt.xlim([1, 512])
plt.xlabel('Number of Bags')
plt.ylabel('Counting Time (log)')
plt.title('Time for Lone and Infinite Cashiers')

legend = plt.legend(loc='upper left')
plt.savefig('P5.png')
# plt.show()