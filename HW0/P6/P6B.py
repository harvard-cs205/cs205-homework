import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import math

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
    wait_time = [ math.pow(10, -6), math.pow(10, -5), math.pow(10, -4),
    math.pow(10, -3), math.pow(10, -2), math.pow(10, -1), math.pow(10, 0) ]

    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        serialTime = time.time()

        for i in range(0, N):
            burnTime(t)
        serialTime = time.time() - serialTime    

        parallelTime = time.time()
        job = N *[t]
        pool.map(burnTime, job)
        parallelTime = time.time() - parallelTime

        # Compute the ratio of these times
        ratio.append(serialTime / parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
