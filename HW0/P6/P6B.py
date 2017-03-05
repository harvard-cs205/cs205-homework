import multiprocessing as mp
import time
import numpy as np
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

    #Here I chose 18 different wait times
    wait_time = [1e-06, 5e-06, 1e-05, 5e-05, 1e-04, 5e-04, 1e-03, 5e-03,
                 1e-02, 5e-02, 1e-01, 5e-01, 0.6, 0.7, 0.8, 0.9, 0.95, 1]

    for t in wait_time:
        # Compute jobs serially and in parallel

        # Compute parallel time
        st = time.time()
        pool.map(burnTime, [t] * N)
        et = time.time()
        parallelTime = et - st

        #Compute serial time
        st = time.time()
        for i in range(N):
            burnTime(t)
        et = time.time()
        serialTime = et - st

        # Compute the ratio of these times
        print("parallelTime {}: serialTime {}:".format(parallelTime, serialTime))
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
