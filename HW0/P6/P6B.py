import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import random as rand
import numpy as np
import sys
# Sleep for t seconds
def burnTime(t):
    time.sleep(t)
    return time.time()
# Main
if __name__ == '__main__':
    N = 16  # The number of jobs
    P = 4   # The number of processes
    f = open('P6.txt','w')
    f.write('Testing')
    # A thread pool of P processes
    pool = mp.Pool(P)

    # Use a variety of wait times
    ratio = []
    wait_time = []
    for i in range(16):
        wait_time.append(rand.uniform(0.000001,1))

    wait_time = np.sort(wait_time)
    parallelStartTime = time.time()
    parallelResult = pool.map(burnTime,wait_time)
    parallelElapsedTime = time.time() - parallelStartTime
    parallelResult = np.array(parallelResult)
    parallelResult -= parallelStartTime
    serialStartTime = time.time()
    serialResult = []
    for t in wait_time:
        temp1 = time.time()
        temp2 = burnTime(t)
        serialResult.append(temp2-temp1)
        serialTime = 1
        parallelTime = 1
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        # Compute the ratio of these times
        # ratio.append(serialTime/parallelTime)
    serialElapsedTime = time.time() - serialStartTime
    #print "serialElapsedTime",serialElapsedTime
    #print "parallelElapsedTime",parallelElapsedTime
    #print "parallelResult=",parallelResult
    #print "serialResult=",serialResult
    # Plot the results
    for i in range(len(serialResult)):
        ratio.append(serialResult[i]/parallelResult[i])
    #print "ratio=",ratio
    #print "Waittime=",wait_time
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    #plt.show()
