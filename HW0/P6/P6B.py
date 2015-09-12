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
    wait_time = np.logspace(-6,0,7);
    print wait_time;

    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        # Compute serial time
        start = time.time();
        for i in range(0,16):
            burnTime(t);
        done = time.time();
        serialTime = done - start;
        
        # Do parallel 
        start = time.time();
        pool.map(burnTime,[t for i in range(16)]);
        done = time.time();

        parallelTime = done - start;

        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime);




    # # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
