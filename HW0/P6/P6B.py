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
    wait_time = [10e-6, 5e-6,10e-5,5e-5,10e-4,5e-4,10e-3,5e-3,10e-2,5e-2,10e-1,5e-1,1]

    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
	# serially
	ts_start = time.time()
	for i in range(16):
	    burnTime(t)
	ts_end = time.time()

        serialTime = ts_end - ts_start

	tp_start = time.time()
	result = pool.map(burnTime, [t]*16)	
	tp_end = time.time()
        parallelTime = tp_end - tp_start

        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)
	
    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
