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

    t = 1
    mult_5 = True
    while t <= 10 ** 6:
        wait_time.append(float(t) / (10 ** 6))
        t = t * [2, 5][mult_5]
        mult_5 = not mult_5

    print wait_time
    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        serialTime = 1
        parallelTime = 1

        # counting time ellapsed in serial
        startT = time.time()
        for i in xrange(N):
            burnTime(t)
        endT = time.time()
        serialTime = endT - startT
        # counting time ellapsed in parallel
        startT = time.time()
        pool.map(burnTime, [t] * N)
        endT = time.time()
        parallelTime = endT - startT
        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
