import multiprocessing as mp
import numpy as np
import time
import matplotlib.pyplot as plt

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
    wait_time = [10 ** (-6)]
    for k in xrange(6):
        wait_time.append(5 * wait_time[-1])
        wait_time.append(2 * wait_time[-1])

    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        start_ite_time = time.time()
        for k in xrange(16):
            burnTime(t)
        serialTime = time.time() - start_ite_time
        start_par_time = time.time()
        pool.map(burnTime, t * np.ones((N,)))
        parallelTime = time.time() - start_par_time

        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
