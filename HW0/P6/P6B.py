import numpy as np
import multiprocessing as mp
import time
import matplotlib.pyplot as plt

# Sleep for t seconds
def burnTime(t):
    time.sleep(t)




# Main
if __name__ == '__main__':
    N = 16  # The number of jobs
    P = 4   # The number of processes

    pool = mp.Pool(P)

    # Use a variety of wait times
    ratio = []
    wait_time = np.linspace(10**-6,10,1000)

    for t in wait_time:
        # Compute jobs serially and in parallel
        pool.map(burnTime,range(N))        
        parallelTime = time.time()        
        
        i = 1
        while i<=N:
            burnTime(t)
            time.time()
        # Use time.time() to compute the elapsed time for each
            serialTime = time.time()
            i+=1

        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
