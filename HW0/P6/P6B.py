import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import random 

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
    # Get 16 random wait times
    wait_time = sorted([random.uniform(.00000000001, .001) for i in range(16)])

    for t in wait_time:
        # Parallel run 
        t0 = time.time()
        # Use the current wait time mapped over the pool of processors
        pool.map(burnTime, [t for i in range(16)])
        parallelTime = time.time() - t0

        # Serial Run
        t0 = time.time()
        for i in range(16):
            burnTime(t)
        serialTime = time.time() - t0
        # Compute jobs serially and in parallel

        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
