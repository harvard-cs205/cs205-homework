import sys
sys.path.append('../util')

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import numpy as np
import matplotlib.pyplot as plt
from timer import Timer
from parallel_vector import move_data_serial, move_data_fine_grained, move_data_medium_grained

if __name__ == '__main__':
    ########################################
    # Generate some test data, first, uncorrelated
    ########################################
    orig_counts = np.arange(1000, dtype=np.int32)
    src = np.random.randint(1000, size=1000000).astype(np.int32)
    dest = np.random.randint(1000, size=1000000).astype(np.int32)

    # Different values of N to test
    N_sample = [10]
    iterations = 1

    total = orig_counts.sum()

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

    # Looking for different values of N
    medium_grained_uncorr = []
    for N in N_sample:
        local_time = []
        for _ in xrange(iterations):
            counts[:] = orig_counts
            with Timer() as t:
                move_data_medium_grained(counts, src, dest, 100, N)
            assert counts.sum() == total, "Wrong total after move_data_medium_grained"
            print("Medium grained uncorrelated: {} seconds".format(t.interval))
            local_time.append(t.interval)
        medium_grained_uncorr.append(local_time)

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
    medium_grained_corr = []
    for N in N_sample:
        local_time = []
        for _ in xrange(iterations):
            counts[:] = orig_counts
            with Timer() as t:
                move_data_medium_grained(counts, src, dest, 100, N)
            assert counts.sum() == total, "Wrong total after move_data_medium_grained"
            print("Medium grained correlated: {} seconds".format(t.interval))
            local_time.append(t.interval)
        medium_grained_corr.append(local_time)

    # Plot the result
    plt.plot(N_sample, [min(e) for e in medium_grained_corr], label='Correlated')
    plt.plot(N_sample, [min(e) for e in medium_grained_uncorr], label='Uncorrelated')

    plt.xlabel('N')
    plt.ylabel('Execution Time')
    plt.title('Medium Grained Execution Time')

    plt.legend(loc=1)
    plt.show()
