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

    # A thread pool of P processes
    pool = mp.Pool(P)

    # Use a variety of wait times
    ratio = []
    wait_time = 10**np.arange(-6, 0.2, .2)

    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        
        # Serial compute
        serial_time_start = time.time()
        for i in xrange(N):
            burnTime(t)
        serial_time_stop = time.time()
        serialTime = serial_time_stop - serial_time_start

        # Parallel compute
        parallel_time_start = time.time()
        result = pool.map(burnTime, N*[t])
        parallel_time_stop = time.time()
        parallelTime = parallel_time_stop - parallel_time_start

        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
