import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import numpy as np
import random

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

    # Generate 16 random wait times
    wait_time = np.logspace(-6, 0, 25)#[random.uniform(10**(-6), 1) for x in range(100)]

    for t in wait_time:
        print 'On wait time', t, '!'
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each

        # Calculate the time it take serially
        serialTime = time.time()
        for ii in range(N):
            burnTime(t)
        serialTime = time.time() - serialTime 

        # Calculate the time it takes in parallel (why are we doing this in a loop)
        parallelTime = time.time()
        pool.map(burnTime, [t for x in range(N)])
        parallelTime = time.time() - parallelTime

        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, 'ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
