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
    wait_time = []

    wait_time = [10.0 ** (ii * 0.5) for ii in range(-6 * 2, 1)]

    for t in wait_time:

        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        serialTime = 1
        parallelTime = 1
        
        ## serial time
        Ts = time.time()
        for i in xrange(N):
            burnTime(t)
        Te = time.time()
        serialTime = Te - Ts
        
        ## paralel time
        Ts = time.time()
        pool.map(burnTime , [t] * N)
        Te = time.time()
        parallelTime = Te - Ts

        ratio.append(serialTime/parallelTime)

        # Compute the ratio of these times
        # ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
