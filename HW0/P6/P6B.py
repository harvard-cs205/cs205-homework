import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import numpy as np

# Sleep for t seconds
def burnTime(t):
    time.sleep(t)

# Main
if __name__ == '__main__':
    N = 16  # The number of jobs
    P = 4   # The number of processes

    # A thread pool of P processes
    pool = mp.Pool(P)

    # Use a variety  f wait times
    ratio = []
    wait_time = [10**(k) for k in range(-6,1,1)]

    for t in wait_time:
        # Compute jobs serially 
        begin=time.time()
        for k in range(16):
            burnTime(t)
        end=time.time()
        # Use time.time() to compute the elapsed time for each
        serialTime = begin-end 
        print(serialTime)
        #and in parallel       
        begin=time.time()
        pool.map(burnTime, range(16))
        end=time.time()
        # Use time.time() to compute the elapsed time for each
        parallelTime = begin-end 

        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
