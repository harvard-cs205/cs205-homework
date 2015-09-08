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

    # Use a variety of wait times
    ratio = []
    wait_time =[1e-6,2*1e-6,5*1e-6,1e-5,1e-5,5*1e-5,7*1e-5,1e-4,5*1e-4,1e-3,
                5*1e-3,7*1e-3,1e-2,5*1e-2,0.1,0.5,1] 
    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        #serially
        i=0
        serialTime=time.time()
        while i<N:
          burnTime(t)
          i=i+1
        serialTime=time.time()-serialTime
        #in parallel
        parallelTime=time.time()
        pool.map(burnTime,[t]*N)
        parallelTime=time.time()-parallelTime

        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.savefig('P6.png')
    plt.show()
