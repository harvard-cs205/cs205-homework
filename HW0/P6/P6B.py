import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import random as rand
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
    for i in range(16):
        wait_time.append(rand.uniform(0.000001,1))

    parallelStartTime = time.time()
    parallelResult = pool.map(burnTime,wait_time)
    parallelElapsedTime = time.time() - parallelStartTime

    serialStartTime = time.time()
    for t in wait_time:
        burnTime(t)
        serialTime = 1
        parallelTime = 1
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        # Compute the ratio of these times
        # ratio.append(serialTime/parallelTime)
    serialElapsedTime = time.time() - serialStartTime
    print serialElapsedTime
    print parallelElapsedTime
    # Plot the results
    # plt.plot(wait_time, ratio, '-ob')
    # plt.xscale('log')
    # plt.xlabel('Wait Time (sec)')
    # plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    # plt.title('Speedup versus function time')
    # plt.show()
