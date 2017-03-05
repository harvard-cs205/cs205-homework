import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import math
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
    # place points at 40 locations between 10^-7 and 10^0
    wait_time = np.logspace(np.log10(pow(10,-6)), np.log10(pow(10,0)), 40)
    print wait_time
    for t in wait_time:
        #print wait_time
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        startTime = time.time()
        #Serial computation
        for i in range(N):
            burnTime(t)
        serialTime = time.time() - startTime

        startTime = time.time()
        #Parallel computation
        result = pool.map(burnTime, [t for x in range(N)])
        parallelTime = time.time() - startTime

        # Compute the ratio of these times]
        print str(t) + " serial: " + str(serialTime) + "      par: " + str(parallelTime)
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
