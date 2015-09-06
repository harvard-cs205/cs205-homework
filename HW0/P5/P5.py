# Sam Daulton
# Sept 6, 2015

import numpy as np
import matplotlib.pyplot as plt
import math


def time_with_infinite_workers(n):
    '''
    Returns the number of seconds it takes to add n numbers, where it takes 1 second to add 2 numbers, and only 2 numbers may be added by a single worker.
    '''
    return math.ceil(math.log(n,2))

def time_with_one_worker(n):
    '''
    Returns the number of seconds it takes to add n numbers, where it takes 1 second to add 2 numbers, and only 2 numbers may be added by a single worker.
    '''
    num_left = n
    total_time = 0
    while (num_left != 1):
        num_left = num_left/2
        total_time += num_left
    return total_time
        
num = range(1,1000)
x = np.array(num)
y1 = np.ceil(np.log2(x))
times = [0]*999
print time_with_one_worker(8)
for i in num:
    times[i-1]=time_with_one_worker(i)
y2 =  np.array(times)
print x
plt.ylim(0,100)
plt.plot(x, y1, 'r--', x, y2, 'bs')
plt.show()
plt.close()
