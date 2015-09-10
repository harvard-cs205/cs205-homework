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
    wait_time = np.logspace(-6, 0, 10)
    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        # serial process
        start_serial = time.time()
        for i in xrange(N):
            burnTime(t)
        serialTime = time.time() - start_serial

        start_parallel = time.time()
        pool.map(burnTime, [t for i in xrange(N)])
        parallelTime = time.time() - start_parallel

        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
