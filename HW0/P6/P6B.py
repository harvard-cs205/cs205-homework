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

    # use 15 wait times from 10^-6 to 10^0
    wait_time = np.logspace(-6, 0, 15)

    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        serialTime = 0
        parallelTime = 0

        # run N times with simple loop
        time_start = time.time()
        for _ in range(N):
            burnTime(t)

        serialTime = time.time() - time_start

        # run N times in parallel
        time_start = time.time()
        wait_times = [t] * N
        pool.map(burnTime, wait_times)

        parallelTime = time.time() - time_start

        # Compute the ratio of these times
        ratio.append(serialTime / parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
