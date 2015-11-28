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
    wait_time = [10**x for x in range(-6,1)]

    for t in wait_time:
        # Compute jobs serially and in parallel
        Tp1 = time.time()
        pool.map(burnTime, [t for _ in range(N)])
        Tp2 = time.time()

        Ts1 = time.time()
        for _ in range(N):
            burnTime(t)
        Ts2 = time.time()

        serialTime = Ts2-Ts1
        parallelTime = Tp2-Tp1

        # Compute the ratio of these times
        ratio.append(serialTime/parallelTime)

    # Plot the results
    plt.plot(wait_time, ratio, '-ob')
    plt.xscale('log')
    plt.xlabel('Wait Time (sec)')
    plt.ylabel('Serial Time (sec) / Parallel Time (sec)')
    plt.title('Speedup versus function time')
    plt.show()
