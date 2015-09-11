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
    iterations = 50 #number of iterations
    lower_bound = 10**(-6) # given lower bound of time
    upper_bound = 10**0 # given upper bound of time

    # A thread pool of P processes
    pool = mp.Pool(P)

    # Use a variety of wait times
    ratio = []
    #wait_time = np.linspace(lower_bound, upper_bound, iterations)
    wait_time = [10**(-6), 10**(-5.5), 10**(-5), 10**(-4.5), 10**(-4), 10**(-3.5), 10**(-3), 10**(-2.5), 10**(-2), 10**(-1.5), 10**(-1), 10**(-.5), 10**(0)]

    for t in wait_time:
        
        start_serial = time.time()
        for j in range(N):
            burnTime(t)
        serialTime = time.time() - start_serial
        
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        #serialTime = 1
        #parallelTime = 1
        
        start_parallel = time.time()
        result = pool.map(burnTime(t), range(N))
        parallelTime = time.time() - start_parallel


        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()