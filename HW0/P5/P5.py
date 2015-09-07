# Sam Daulton
# Sept 6, 2015

import numpy as np
import matplotlib.pyplot as plt
import math

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
        
x_range = range(1,1000)
x = np.array(x_range)
y1 = np.ceil(np.log2(x))
times = [0]*999
for i in x_range:
    times[i-1]=time_with_one_worker(i)
y2 =  np.array(times)
plt.ylim(0,100)
plt.plot(x, y1, 'r--', label="With infinite workers")
plt.plot(x,y2,'g--',label="With one worker")
plt.ylabel("Time (seconds)")
plt.xlabel("# of Workers")
plt.legend();
#plt.show()
plt.savefig("P5.png")
plt.close()

