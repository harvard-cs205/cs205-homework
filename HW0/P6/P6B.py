import multiprocessing as mp
import numpy as np
import time
import matplotlib.pyplot as plt
from pylab import savefig

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
    wait_time = 10**np.linspace(-6,0,20)


    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        timeBegining = time.time()

        pool.map(burnTime, t*np.ones(N))

        timeEnd = time.time()

        parallelTime = timeBegining-timeEnd

        timeBegining = time.time()

        for ii in range(N):
            burnTime(t)

        timeEnd = time.time()

        serialTime = timeBegining-timeEnd
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    savefig('P6.png', bbox_inches='tight')
    plt.show()
