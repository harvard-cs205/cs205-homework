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

    pool = mp.Pool(P) #create a pool of P processes

    # Use a variety of wait times
    ratio = [] #empty ratio array to be filled in below
    wait_time = np.array([10**-6,10**-5,10**-4,10**-3,10**-2,.5*10**-1,10**-1,1])

    for t in wait_time:
        # Compute jobs serially and in parallel
        pool.map(burnTime(t),range(N))        
        parallelTime = time.time()        
        
        i = 1
        while i<=N:
            burnTime(t)
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
