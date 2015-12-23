import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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
    # average and standard error of the mean arrays
    ratio_mean = []
    ratio_err = []
    wait_time = 10**(np.arange(0,13)*-0.5)
    
    # repetitions for statistical meaning
    rep = 5
    
    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        ratio = []
        for y in range(rep):
            # serial processing
            x = np.arange(0,N)
            serialTime = time.time()
            for i in x:
                burnTime(t)
            serialTime = time.time() - serialTime
            
            # parallel processing
            x = np.ones(N)*t
            parallelTime = time.time()
            result = pool.map(burnTime,x)
            parallelTime = time.time() - parallelTime
            
            # Compute the ratio of these times
            ratio.append(serialTime/parallelTime)
        
        # determine the statistical stuff here
        ratio_mean.append(np.mean(ratio))
        ratio_err.append(stats.sem(ratio))

    # Plot the results
    plt.plot(wait_time, ratio_mean, '-ob')
    plt.errorbar(wait_time,ratio_mean,yerr=ratio_err)
    plt.plot(wait_time,np.ones(len(wait_time)),'--g')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
