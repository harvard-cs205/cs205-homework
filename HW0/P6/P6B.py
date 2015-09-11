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
    wait_time = [10**(-i) for i in range(7)]

    for t in wait_time:
        # Compute jobs serially and in parallel
	serialTime_start = time.time()
	for i in range(N):
		burnTime(t)
	serialTime_end = time.time()
	
	parallelTime_start = time.time()
	pool.map(burnTime, [t]*N)
	parallelTime_end = time.time()
	
        # Use time.time() to compute the elapsed time for each
        serialTime = serialTime_end - serialTime_start
        parallelTime = parallelTime_end - parallelTime_start

        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
