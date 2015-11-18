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

    correlated = []
    uncorrelated = []
    N_list = [5, 10, 20, 30, 50, 75, 100]    

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
    uncorrelated.append(t.interval)

    ########################################
    # You should explore different values for the number of locks in the medium
    # grained locking
    ########################################
    counts[:] = orig_counts

    for N in N_list:
        minimum = 100
        for i in range(3):
            with Timer() as t:
                move_data_medium_grained(counts, src, dest, 100, N)
            assert counts.sum() == total, "Wrong total after move_data_medium_grained"
            print("Medium grained uncorrelated: {} seconds".format(t.interval))
            minimum = min(minimum, t.interval)
        uncorrelated.append(minimum)

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
    correlated.append(t.interval)

    ########################################
    # You should explore different values for the number of locks in the medium
    # grained locking
    ########################################
    counts[:] = orig_counts

    for N in N_list:
        minimum = 100
        for i in range(3):
            with Timer() as t:
                move_data_medium_grained(counts, src, dest, 100, N)
            assert counts.sum() == total, "Wrong total after move_data_medium_grained"
            print("Medium grained correlated: {} seconds".format(t.interval))
            minimum = min(t.interval, minimum)
        correlated.append(minimum)

    plt.plot([1] + N_list, uncorrelated)
    plt.plot([1] + N_list, correlated)
    plt.xlabel('N')
    plt.ylabel('Run time')
    plt.title('Comparing Correlated and Uncorrelated Run times')
    plt.legend(['Uncorrelated', 'Correlated'], loc='upper left')
    plt.savefig('myfig2')
    plt.show()
