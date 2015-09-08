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
    wait_time = np.logspace(-6, 0, 20) # wait time from 10e-6 to 10e0
    print(wait_time)

    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each

        # Compute the ratio of these times
        # ratio.append(serialTime/parallelTime)

        # Serial Timing
        start_s = time.time()
        for i in range(N):
            burnTime(t)
        serialTime = time.time() - start_s

        # Parallel Timing
        start_p = time.time()
        for i in range(N/4):
            pool.apply(burnTime, (t,)) # each loop runs 4 burnTime processes
        parallelTime = time.time() - start_p

        # Append Ratio
        ratio.append(serialTime/parallelTime*1.0)

        print(serialTime)
        print(parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()