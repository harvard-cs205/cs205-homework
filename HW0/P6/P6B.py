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

    # Initialize list for serial vs parallel ratios
    ratio = []    
    
    # Range of wait times between 10^-6 and 10^0
    wait_time = [y*10**x for x in range(-6,0) for y in range (1,10)] + [1]
    
    for t in wait_time:
        
        # Compute jobs serially
        start = time.time()
        for job in range(N):
            burnTime(t)
        serialTime = time.time() - start
        
        # Compute jobs in parallel
        start = time.time()
        pool.map(burnTime, [t for x in range(N)])
        parallelTime = time.time() - start

        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
