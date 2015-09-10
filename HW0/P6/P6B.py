import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import numbers
import numpy as np

# Sleep for t seconds
def burnTime(t):
    time.sleep(t)

# serial run
def serialRun(t):
    for i in range(1,16):
        burnTime(t)

# parallel run
def parallelRun(t):
    assert isinstance(t, numbers.Number), "t must be a number"
    pool = mp.Pool(4)
    pool.map(burnTime, [t] * 16) # repeat sleeping time 16 times and transform it to list, be careful that t is a number!

# Main
if __name__ == '__main__':
    N = 16  # The number of jobs
    P = 4   # The number of processes

    # A thread pool of P processes
    pool = mp.Pool(P)

    # Use a variety of wait times
    ratio = []
    wait_time = (10**np.linspace(-6, 0, num=30)).tolist()

    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        serialTime = 1
        parallelTime = 1

        print('running 16 jobs for t = ' + str(t))

        # run serially
        serialTime = time.time()
        serialRun(t)
        serialTime = time.time() - serialTime

        # run parallel
        parallelTime = time.time()
        parallelRun(t)
        parallelTime = time.time() - parallelTime

        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.savefig('P6.png', format='png')
    plt.show()
