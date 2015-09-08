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
    ratio = []
    wait_time = np.logspace(-6, 0, 20)

    for t in wait_time:
         
         # Serial Loop
         time_start = time.time()
         for ii in range(N):
              burnTime(t)
         serialTime = time.time() - time_start

		 #Parallel loop
         time_start = time.time()
         for jj in range(N):
             results = pool.map(burnTime(t), range(P))
         parallelTime = time.time() - time_start

         # Compute the ratio of these times
         ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
