import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import random as rand
import numpy as np
import sys
import pdb
# Sleep for t seconds
def burnTime(t):
    time.sleep(t)
    return
# Main
if __name__ == '__main__':
    N = 16  # The number of jobs
    P = 4   # The number of processes
    # f = open('P6.txt','a')
    
    # A thread pool of P processes
    pool = mp.Pool(P)

    # Use a variety of wait times
    ratio = []
    wait_time = [0.000001,0.00001,0.0001,0.001,0.01,0.1]
    # for i in range(10):
    #     wait_time.append(rand.uniform(0.000001,1))

    #wait_time = np.sort(wait_time)
    parallelTimes = []
    serialTimes = []
    #pdb.set_trace()
    for i in range(len(wait_time)):
        start1 = time.time()
        for j in range(16):
            burnTime(wait_time[i])
        serialTimes.append(time.time() - start1)
    for i in range(len(wait_time)):
        start = time.time()
        parallelResult = pool.map(burnTime,[wait_time[i]]*16)
        parallelTimes.append(time.time()-start)
    
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        # Compute the ratio of these times
        # ratio.append(serialTime/parallelTime)

    #print "serialElapsedTime",serialElapsedTime
    #print "parallelElapsedTime",parallelElapsedTime
    #print "parallelResult=",parallelResult
    #print "serialResult=",serialResult
    # f.write('Serial Results = [')
    # for x in serialResult:
    #     f.write("%s " % x)
    # f.write(']\n')
    # f.write('Parallel Results = [')
    # for x in parallelResult:
    #     f.write("%s " % x)
    # f.write(']\n')
    # f.write('Wait times = [')
    # for x in wait_time:
    #     f.write("%s " % x)
    # f.write(']\n')
    
    # Plot the results
    
    # f.write('Ratios = [')
    # for x in ratio:
    #     f.write("%s " % x)
    # f.write(']\n')
    #print "ratio=",ratio
    ratios = []
    for i in range(len(serialTimes)):
        ratios.append(serialTimes[i]/parallelTimes[i])
    print 'serialTimes',serialTimes
    print 'parallelTimes',parallelTimes
    print 'ratios',ratios
    #print "Waittime=",wait_time
    plt.plot(wait_time, ratios, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.savefig('P6B.png')
    plt.show()
