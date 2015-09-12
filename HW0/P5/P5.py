import matplotlib.pyplot as plt
import math

x = range(1, 512)
y_inf = [math.log(i, 2) for i in x]
y_alone = [i-1 for i in x]

plt.plot(x, y_inf, '-b', label='Infinite Employees')
plt.plot(x, y_alone, '--g', label='Lone Cashier')
plt.xlabel('Number of Bags')
plt.ylabel('Counting Time')
plt.title('')

legend = plt.legend(loc='upper left')
plt.savefig('P5.png')
# plt.show()