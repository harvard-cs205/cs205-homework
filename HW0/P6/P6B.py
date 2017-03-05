import multiprocessing as mp
import numpy as np
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
    wait_time = np.logspace(-10, 0, num=500) 

    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        start = time.time()
        pool.map(burnTime, [t]*N, chunksize=1)
        parallelTime = start - time.time()
        
        start = time.time()
        map(burnTime, [t]*N)
        #[burnTime(t) for i in range(N)]
        serialTime = start - time.time()
        
     
        # Compute the ratio of these times
        ratio_t = serialTime/parallelTime
        ratio.append(ratio_t)
        

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.axhline(1, color='r', linestyle='--', label="Serial = Parallel")
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.legend(loc='best')
    plt.show()
