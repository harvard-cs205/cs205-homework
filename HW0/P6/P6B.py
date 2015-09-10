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
    #wait_time = []
    wait_time = np.logspace(-6, 0, num=20)
    # print wait_time
    for t in wait_time:
        print "Current wait time: ", t
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        serialTime = 1
        parallelTime = 1
        startSerialTime = time.time()
        # Insert serial code here
        for i in range(N):
            burnTime(t)
        serialTime = time.time() - startSerialTime

        startParellelTime = time.time()
        # Insert parellel code here
        pool.map(burnTime, N*[t])
        parallelTime = time.time() - startParellelTime
        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
