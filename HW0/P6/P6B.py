
# coding: utf-8

# In[1]:

import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt

# Sleep for t seconds
def burnTime(t):
    time.sleep(t)


# In[2]:

# Main
if __name__ == '__main__':
    N = 16  # The number of jobs
    P = 4   # The number of processes

    # A thread pool of P processes
    pool = mp.Pool(P)

    # Use a variety of wait times
    ratio = []
    a=range(-6, 1)
    wait_time = [10**i for i in a]
    
    def serial_Time(t):
        start_time1 = time.time()
        for i in range(1, N+1):
            burnTime(t)
        
        end_time1=time.time()
        serialTime = end_time1- start_time1
        return serialTime
    
    def parallel_Time(t):
        start_time2 = time.time()
        pool.map(burnTime, [t]*N)
        end_time2=time.time()
        parallelTime = end_time2- start_time2
        return parallelTime      

    # Compute the ratio of these times
    ratio=[serial_Time(t)/parallel_Time(t) for t in  wait_time]
    print ratio
    
    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()


# In[ ]:




# In[ ]:



