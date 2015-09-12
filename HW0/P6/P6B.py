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
    wait_time = [10**(-i) for i in xrange(7)]

    for t in wait_time:

        # Compute job in parallel
        start_parallelTime = time.time()
        result = pool.map(burnTime, [t]*N)
        stop_parallelTime = time.time()

        # Compute job in serial
        start_serialTime = time.time()
        for i in xrange(N):
            burnTime(t)
        stop_serialTime = time.time()

        # Compute the ratio of these times
        parallelTime = stop_parallelTime - start_parallelTime
        serialTime = stop_serialTime - start_serialTime
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
