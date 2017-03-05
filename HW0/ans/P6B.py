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
    wait_time = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1]

    def s_time(t,n):
        t1=time.time()
	for i in range(n):
	    burnTime(t)
        t2=time.time()
   	return t2-t1
    def p_time(t,n):
	t1=time.time()
	pool.map(burnTime, [t]*n)
	t2=time.time()
        return t2-t1

    # Compute the ratio of these times
    ratio=[s_time(t,N)/p_time(t,N)  for t in wait_time]

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
