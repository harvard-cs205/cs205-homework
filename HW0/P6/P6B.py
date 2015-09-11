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
    wait_time = [1e-6,5e-6,1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1,5e-1,1]

    for t in wait_time:
        # Compute jobs serially and in parallel
        # serially
	start_serial = time.time()
	for i in range(N):
          burnTime(t)
	end_serial = time.time()
        # in parallel
	start_parallel = time.time()
 	pool.map(burnTime, N*[t])
	end_parallel = time.time()
        # Use time.time() to compute the elapsed time for each
        serialTime = end_serial-start_serial
        parallelTime = end_parallel-start_parallel
        print serialTime
        print parallelTime
        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.savefig('P6.png')
    plt.show()
