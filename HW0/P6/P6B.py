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
    wait_time = 10.**np.arange(-6,0.5,0.5)

    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        start = time.time()
        for i in range(N): burnTime(t)
        serialTime = time.time() - start

        start = time.time()
        pool.map(burnTime, [t]*N)
        parallelTime = time.time() - start

        # Compute the ratio of these times
        ratio.append(np.exp(np.log(serialTime)-np.log(parallelTime)))

    for t, r in zip(wait_time, ratio):
        print 'Wait Time: ' + str(t)
        print 'Ratio:     ' + str(r)
        print

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.axhline(y=1, c='r', ls='--')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.savefig('P6.png')
    plt.show()
