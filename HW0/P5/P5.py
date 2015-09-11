import matplotlib.pyplot as plt
import numpy as np 

num_bags = []
serial_time = []
parallel_time = []
for n in range(1,500):
#sys.maxint)
    # Plot the results
    num_bags.append(n)
    serial_time.append(n-1)

    parallel_time_calc = 0
    num = n

    while (num >= 2):
        parallel_time_calc += 1
        num = num / 2
    if (num == 1):
        num = num - 1
        parallel_time_calc += 1 

    parallel_time.append(parallel_time_calc)

plt.plot(num_bags, serial_time, color = 'red', label='Single Cashier')
plt.plot(num_bags, parallel_time, color = 'blue', label = 'Infinite Cashiers')
plt.xlabel('Number of Bags')
plt.ylabel('Time to Sum (sec)')
plt.title('Number of Bags vs. Time to Verify Total')
plt.legend()
plt.show()
