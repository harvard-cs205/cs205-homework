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
    wait_time = 10**np.linspace(-6,0,40)

    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        serialTime = 1
        parallelTime = 1

        # in serial
        serialTime = time.time()
        for i in range(N):
            burnTime(t)
        serialTime = time.time() - serialTime


        # in parallel
        # Apply burnTime to this list of "job numbers" using the pool
        parallelTime = time.time()
        p_result = pool.map(burnTime, t*np.ones(N))
        parallelTime = time.time() - parallelTime

        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)




    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
