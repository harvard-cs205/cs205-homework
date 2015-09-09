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
    f = open('P6.txt','r+')
    # A thread pool of P processes
    pool = mp.Pool(P)

    # Use a variety of wait times
    ratio = []
    wait_time = [10**i for i in range(-6,1)]

    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        x = time.time()
        pool.map(burnTime,[t for v in range(16)])

        parallelTime = time.time()-x
        f.write('parallel time for wait_time %fs is %fs\n'%(t,parallelTime))
        x = time.time()
        for k in range(N):
            burnTime(t)
        serialTime = time.time()-x
        f.write('serial time for wait_time %fs is %fs\n'%(t,serialTime))
        #y = time.time()
        #serialTime = burnTime(t)

        # Compute the ratio of these times
        f.write('ratio for wait_time %fs is %fs\n'%(t,serialTime/parallelTime))
        ratio.append(serialTime/parallelTime)
    
    # Plot the results
    f.close()
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.savefig('P6.png')
    plt.show()
