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
    wait_time = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1]

    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        serialTime = 1
        parallelTime = 1

        time0 = time.time()
        #for i in range(N):
        #burnTime(t)
        map(burnTime, [t]*N)
        serialTime = time.time()-time0

        time0 = time.time()
        p = pool.map(burnTime, [t]*N)
        parallelTime = time.time()-time0
        
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
