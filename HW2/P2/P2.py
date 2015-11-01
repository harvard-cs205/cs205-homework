import sys
sys.path.append('../util')

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import numpy as np
from timer import Timer
from parallel_vector import move_data_serial, move_data_fine_grained, move_data_medium_grained

import matplotlib.pyplot as plt

if __name__ == '__main__':
    ########################################
    # Generate some test data, first, uncorrelated
    ########################################
    orig_counts = np.arange(1000, dtype=np.int32)
    src = np.random.randint(1000, size=1000000).astype(np.int32)
    dest = np.random.randint(1000, size=1000000).astype(np.int32)

    total = orig_counts.sum()

    # data for plots
    lock_times = [1, 5, 10, 20, 50, 100]
    trials = 5
    num_locks = len(lock_times)
    uncorrelated_times = [sys.maxint]*num_locks
    correlated_times = [sys.maxint]*num_locks


    # serial move
    counts = orig_counts.copy()
    with Timer() as t:
        move_data_serial(counts, src, dest, 100)
    assert counts.sum() == total, "Wrong total after move_data_serial"
    print("Serial uncorrelated: {} seconds".format(t.interval))
    serial_counts = counts.copy()

    # fine grained
    counts[:] = orig_counts
    with Timer() as t:
        move_data_fine_grained(counts, src, dest, 100)
    assert counts.sum() == total, "Wrong total after move_data_fine_grained"
    print("Fine grained uncorrelated: {} seconds".format(t.interval))

    ########################################
    # You should explore different values for the number of locks in the medium
    # grained locking
    ########################################
    for x in xrange(trials):
        for i in xrange(num_locks):
            N = lock_times[i]
            counts[:] = orig_counts
            with Timer() as t:
                move_data_medium_grained(counts, src, dest, 100, N)
            assert counts.sum() == total, "Wrong total after move_data_medium_grained"
            uncorrelated_times[i] = min(uncorrelated_times[i], t.interval)
    for i in xrange(num_locks):
        N = lock_times[i]
        time = uncorrelated_times[i]
        print("Medium grained uncorrelated for N={}: {} seconds".format(N, time))

    ########################################
    # Now use correlated data movement
    ########################################
    dest = src + np.random.randint(-10, 11, size=src.size)
    dest[dest < 0] += 1000
    dest[dest >= 1000] -= 1000
    dest = dest.astype(np.int32)

    # serial move
    counts[:] = orig_counts
    with Timer() as t:
        move_data_serial(counts, src, dest, 100)
    assert counts.sum() == total, "Wrong total after move_data_serial"
    print("Serial correlated: {} seconds".format(t.interval))
    serial_counts = counts.copy()

    # fine grained
    counts[:] = orig_counts
    with Timer() as t:
        move_data_fine_grained(counts, src, dest, 100)
    assert counts.sum() == total, "Wrong total after move_data_fine_grained"
    print("Fine grained correlated: {} seconds".format(t.interval))

    ########################################
    # You should explore different values for the number of locks in the medium
    # grained locking
    ########################################
    for x in xrange(trials):
        for i in xrange(num_locks):
            N = lock_times[i]
            counts[:] = orig_counts
            with Timer() as t:
                move_data_medium_grained(counts, src, dest, 100, N)
            assert counts.sum() == total, "Wrong total after move_data_medium_grained"
            correlated_times[i] = min(correlated_times[i], t.interval)
    for i in xrange(num_locks):
        N = lock_times[i]
        time = correlated_times[i]
        print("Medium grained correlated for N={}: {} seconds".format(N, time))

    plt.plot(lock_times, uncorrelated_times)
    plt.plot(lock_times, correlated_times)
    plt.title('Uncorrelated and correlated data movement time for N locks')
    plt.xlabel('N = # of locks')
    plt.ylabel('Time')
    plt.legend(['Uncorrelated times', 'Correlated times'])
    plt.show()