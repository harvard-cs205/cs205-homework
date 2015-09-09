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
    ratio =np.zeros(7)
    parallelTime=np.zeros(7)
    serialTime=np.zeros(7)
#    wait_time = 10**(-6)*np.array(range(1,10**2+1))
    wait_time=[10**(-6),10**(-5),10**(-4),10**(-3),10**(-2),10**(-1),10**(0)]
    i=0
    for t in wait_time:
        # Compute jobs serially and in parallel
		time_p1=time.time()
		results_parallel=pool.map(burnTime,t*(np.ones(16)))
		time_p2=time.time()
		parallelTime[i] = time_p2-time_p1
		time_s1=time.time()
		for j in range(0,16):
			burnTime(t)

		time_s2=time.time()
		serialTime[i] = time_s2-time_s1
		i+=1

    time_p=np.array(parallelTime)
    time_s=np.array(serialTime)
    ratio=time_s/time_p


        # Compute the ratio of these times
        # ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
