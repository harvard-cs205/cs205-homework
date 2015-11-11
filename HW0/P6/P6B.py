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
    wait_time = np.logspace(-6, 0, 20)

    for t in wait_time:
        # Run the jobs serially
        serialTime = time.time()
        for job in range(N):
            burnTime(t)
        serialTime = time.time() - serialTime

        # Run the jobs in parallel
        parallelTime = time.time()
        pool.map(burnTime, [t]*N)
        parallelTime = time.time() - parallelTime

        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)
        print ratio

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.savefig('P6.png')
    plt.show()