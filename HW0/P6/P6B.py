import multiprocessing as mp
import numpy as np
import time
import matplotlib.pyplot as plt

# Sleep for t seconds
def burnTime(t):
    time.sleep(t)

def getSerialTime(waitTime,numIt):
    startT = time.time()
    for ii in range(0,numIt):
    	burnTime(waitTime)
    endT = time.time()
    return endT-startT

def getParallelTime(waitTime,pool,numIt):
    startT = time.time()
    pool.map(burnTime, np.array([waitTime]*numIt))
    endT = time.time()
    return endT-startT


# Main
if __name__ == '__main__':
    N = 16  # The number of jobs
    P = 4   # The number of processes

    # A thread pool of P processes
    pool = mp.Pool(P)

    # Use a variety of wait times
    wait_time = [10**-6,5*10**-6,10**-5,5*10**-5,10**-4,5*10**-4,10**-3,5*10**-3,10**-2,5*10**-2,10**-1,.5,1]
    ratio = []
    count = 0

    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        serialTime = getSerialTime(t,N)
        parallelTime = getParallelTime(t,pool,N)

        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
    print ratio
