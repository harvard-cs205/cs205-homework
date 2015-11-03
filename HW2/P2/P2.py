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

    total = orig_counts.sum()

    N_values = [2, 5, 10, 50, 100, 1000]
    N_values = [2, 5]

    # serial move
    counts = orig_counts.copy()
    with Timer() as t:
        move_data_serial(counts, src, dest, 100)
    assert counts.sum() == total, "Wrong total after move_data_serial"
    print("Serial uncorrelated: {} seconds".format(t.interval))
    serial_uncorrelated = t.interval
    serial_counts = counts.copy()

    # fine grained
    counts[:] = orig_counts
    with Timer() as t:
        move_data_fine_grained(counts, src, dest, 100)
    assert counts.sum() == total, "Wrong total after move_data_fine_grained"
    fine_grained_uncorrelated = t.interval
    print("Fine grained uncorrelated: {} seconds".format(t.interval))

    ########################################
    # You should explore different values for the number of locks in the medium
    # grained locking
    ########################################
    coarse_uncorrelated = {}
    for N in N_values:
        counts[:] = orig_counts
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, N)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        print("Medium grained uncorrelated (N={}): {} seconds".format(N, t.interval))
        coarse_uncorrelated[N] = t.interval

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
    serial_correlated = t.interval
    serial_counts = counts.copy()

    # fine grained
    counts[:] = orig_counts
    with Timer() as t:
        move_data_fine_grained(counts, src, dest, 100)
    assert counts.sum() == total, "Wrong total after move_data_fine_grained"
    print("Fine grained correlated: {} seconds".format(t.interval))
    fine_grained_correlated = t.interval


    ########################################
    # You should explore different values for the number of locks in the medium
    # grained locking
    ########################################
    coarse_correlated = {}
    for N in N_values:
        counts[:] = orig_counts
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, N)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        print("Medium grained correlated (N={}): {} seconds".format(N, t.interval))
        coarse_correlated[N] = t.interval

    # display results
    values = [serial_correlated, fine_grained_correlated] + coarse_correlated.values()
    labels = ['Serial', 'Fine Grained'] + ['Coarse (N=%d)' % n for n in coarse_correlated.keys()]
    positions = np.arange(len(values))
    plt.bar(positions, values, align='center')
    plt.xticks(positions, labels)
    plt.ylabel('Time to move data')
    plt.title('Correlated Data Movement Times')
    plt.show()

    values = [serial_uncorrelated, fine_grained_uncorrelated] + coarse_uncorrelated.values()
    labels = ['Serial', 'Fine Grained'] + ['Coarse (N=%d)' % n for n in coarse_uncorrelated.keys()]
    positions = np.arange(len(values))
    plt.bar(positions, values, align='center')
    plt.xticks(positions, labels)
    plt.ylabel('Time to move data')
    plt.title('Uncorrelated Data Movement Times')

    plt.show()

