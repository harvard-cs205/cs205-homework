import matplotlib as mpl
mpl.use('Qt4Agg')
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_context('poster', font_scale=1.25)

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
    wait_time = np.logspace(-6, 0, 15)

    serial_time = []
    parallel_time = []

    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each

        # Serial job first
        start_time = time.time()
        for i in range(N):
            burnTime(t)
        time_elapsed = time.time() - start_time
        serial_time.append(time_elapsed)

        # Now parallel job
        start_time = time.time()
        for i in range(N/4): # Each time the pool is run, burntime is called P times!
            pool.apply(burnTime, args=(t,))
        time_elapsed = time.time() - start_time
        parallel_time.append(time_elapsed)

    serial_time = np.array(serial_time)
    parallel_time = np.array(parallel_time)

    # Compute the ratio of these times
    ratio = serial_time/parallel_time

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.savefig('P6.png', dpi=200, bbox_inches='tight')
    plt.show()

