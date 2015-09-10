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
    wait_time = [10.0 ** (ii * 0.5) for ii in range(-6 * 2, 1)]

    for t in wait_time:

        # Compute jobs serially and in parallel
        ts_start = time.time()
        for ii in range(16):
            burnTime(t)
        ts_end = time.time()
        
        tp_start = time.time()
        pool = mp.Pool(4)
        pool.map(burnTime, [t for ii in range(16)])
        tp_end = time.time()

        # Use time.time() to compute the elapsed time for each
        serialTime = ts_end - ts_start
        parallelTime = tp_end - tp_start

        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
