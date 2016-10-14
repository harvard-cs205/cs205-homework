###############################

# CS 205 Fall 2015 Homework 0 Problem 6
# Submitted by Kendrick Lo (Harvard ID: 70984997)
# Github username: ppgmg

import numpy as np
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
    number_samples = 50
    wait_time = np.logspace(-6, 0, number_samples)

    for t in wait_time:

        # Compute jobs serially
        serial_time = time.time()  # initialize with current time
        for i in range(N):
            burnTime(t)    
        serial_time = time.time() - serial_time

        # Compute jobs in parallel
        parallel_time = time.time()  # initialize with current time
        # Call burnTime with argument t, N times using the pool
        pool.map(burnTime, np.repeat(t,N))
        parallel_time = time.time() - parallel_time

        # Compute the ratio of these times
        ratio.append(serial_time/parallel_time)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
