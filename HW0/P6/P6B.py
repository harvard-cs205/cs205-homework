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

    # start = time.time()
    # time.sleep(10)
    # end = time.time()
    # print end - start

    for t in wait_time:
        # Compute jobs serially and in parallel
        startSer = time.time()
        burnTime(t)
        endSer = time.time()

        startPar = time.time()
        pool.map(burnTime,wait_time)
        endPar = time.time()

        # Use time.time() to compute the elapsed time for each
        serialTime = endSer - startSer
        parallelTime = endPar - startPar
        print serialTime
        print parallelTime

        # Compute the ratio of these times
        ratio.append(serialTime / parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()