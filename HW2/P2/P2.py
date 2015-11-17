import sys
sys.path.append('../util')

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()
import matplotlib.pyplot as plt

import numpy as np
from timer import Timer
from parallel_vector import move_data_serial, move_data_fine_grained, move_data_medium_grained

if __name__ == '__main__':
    ########################################
    # Generate some test data, first, uncorrelated
    ########################################
    orig_counts = np.arange(1000, dtype=np.int32)
    src = np.random.randint(1000, size=1000000).astype(np.int32)
    dest = np.random.randint(1000, size=1000000).astype(np.int32)

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
    N = [5, 10, 20, 30, 50, 100]
    counts[:] = orig_counts

    medium_uncorrelated_result = []
    for i in N:
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, i)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        print("Medium grained uncorrelated, N = {}: {} seconds".format(i, t.interval))
        medium_uncorrelated_result.append(t.interval)
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
    # N = 10
    counts[:] = orig_counts
    medium_correlated_result = []
    for i in N:
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, i)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        print("Medium grained correlated, N = {}: {} seconds".format(i, t.interval))
        medium_correlated_result.append(t.interval)

    plt.plot(N, medium_uncorrelated_result, label = 'Medium grained uncorrelated')
    plt.plot(N, medium_correlated_result, label = 'Medium grained correlated')
    plt.title('Comparison Between Correlated and Uncorrelated Transfers')
    plt.legend(loc='upper left')
    plt.show()