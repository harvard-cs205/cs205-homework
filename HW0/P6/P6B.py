import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import numpy as np

# Sleep for t seconds
def burnTime(t):
    time.sleep(t)
    print(t)
# Main
if __name__ == '__main__':
    N = 16  # The number of jobs
    P = 4   # The number of processes

    # A thread pool of P processes
    pool = mp.Pool(P)

    # Use a variety of wait times
    ratio = []
    times = np.logspace(-6,1)
    wait_time = np.ndarray.tolist(times)
    parallelTime = []
    serialTime = []

    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each

        start = time.time()
        time.sleep(t)
        end = time.time()
        serialTime.append(end-start)
        
        start = time.time()
        x = pool.map(burnTime,[t])
        end = time.time()
        parallelTime.append(end-start)

        # Compute the ratio of these times
        ratio.append(serialTime[-1]/parallelTime[-1])

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
