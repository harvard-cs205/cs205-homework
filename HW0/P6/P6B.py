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
    wait_time = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

    i = 0 
    for t in wait_time:
        i += 1
        print "Doing task %d of %d, (%2d %%)" % (i, len(wait_time), 100 * float(i) / len(wait_time))
        inputs = [t for _ in range(N)]
 
        start = time.time() 
        pool.map(burnTime, inputs)
        parallelTime = time.time() - start 
        
        start = time.time()
        for _ in range(N): 
            burnTime(t)
        serialTime = time.time() - start 

        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
