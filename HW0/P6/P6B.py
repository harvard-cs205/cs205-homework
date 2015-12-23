%matplotlib inline

import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import numpy as np

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
    ratio =[]
    wait_time = np.logspace(10E-7,1,num=10,base=10E-7)

    # Serial Process
    for t in wait_time:
        
        starttime=time.time()
        for n in range(N):
            burnTime(t)
        endtime=time.time()
        serialTime = endtime-starttime
            
        starttimep=time.time()
        result = pool.map(burnTime, [t for n in range(N)])
        endtimep=time.time()
        parallelTime=endtimep-starttimep
        
        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)
        
    # Plot the results
plt.plot(wait_time, ratio, '-ob')
plt.xscale('log')
plt.xlabel('Wait Time (sec)')
plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
plt.title('Speedup versus function time')
plt.show()
