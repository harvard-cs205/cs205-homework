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
    wait_time = np.logspace(-6, 0)
    
    for t in wait_time:
        #### Compute the elapsed time for serial time
        start_serial = time.time() # Start the clock
        
        x = 0 # counter for the while loop

        while x < N:
            burnTime(t)
            x += 1
                
        stop_serial = time.time()  # Stop the clock
        serialTime = stop_serial - start_serial  #Subtract to find elapsed time

        
        #### Compute the elapsed time for parallel time
        start_parall = time.time() # Start the clock

        pool.map(burnTime, [t]*N)

        stop_parall = time.time() # Stop the clock

        parallelTime = stop_parall - start_parall

        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
