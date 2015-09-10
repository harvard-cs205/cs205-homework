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
    wait_time = []
    wait_time_ = [10**i for i in range(-6,1)]
    for n in wait_time_[:-2]:
        for i in xrange(1,8):
            wait_time.append(i*n)
    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        x = time.time()
        pool.map(burnTime,[t]*N)
        parallelTime = time.time()-x
        x = time.time()
        for k in range(N):
            burnTime(t)
        serialTime = time.time()-x
        #y = time.time()
        #serialTime = burnTime(t)

        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)
    
    # Plot the result
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.savefig('P6.png')
    plt.show()
