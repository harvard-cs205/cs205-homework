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
    Time = 16
    # A thread pool of P processes
    pool = mp.Pool(P)

    # Use a variety of wait times
    ratio = []
    wait_time = [2e-6, 7e-6, 5e-5, 9e-5, 1e-4 ,6e-4 , 3e-3, 2e-2, 6e-1, 1 ]

    for t in wait_time:
        # Compute jobs serially and in parallel
        # Use time.time() to compute the elapsed time for each
        serialTime = 1
        parallelTime = 1
        
        #parallel time
        pwake = time.time()
        
        results = pool.map(burnTime, [t]*Time)
        
        psleep = time.time()
        parallelTime = pwake-psleep        
        
        #serial time
        swake = time.time()
        
        for x in range(Time):
            burnTime(t)
            
        ssleep = time.time()
        serialTime = swake-ssleep

        # Compute the ratio of these times
        #ratio.append(serialTime/parallelTime)
        ratio.append(serialTime/parallelTime)

 
    

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
