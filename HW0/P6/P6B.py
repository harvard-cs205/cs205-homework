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
    wait_time = [10**x for x in range(-6, 1)] 
    n_iter = 5
    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each

        # Parallel
        t_p = []
        for iter_i in range(n_iter):
            t1_p = time.time()
            pool.map(burnTime(t), range(N))
            t2_p = time.time()
            t_p.append(t2_p - t1_p)

        # Seriali
        t_s = []
        for iter_i in range(n_iter):
            t1_s = time.time()
            for job_i in range(N):
                burnTime(t)
            t2_s = time.time()
            t_s.append(t2_s - t1_s)

        # Compute the ratio of these times
        serialTime = np.mean(t_s)
        parallelTime = np.mean(t_p)
        ratio.append(serialTime/parallelTime)
        print([serialTime, parallelTime])
    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.savefig('P6.png')
