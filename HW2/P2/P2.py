import sys
sys.path.append('../util')
import matplotlib.pyplot as plt
import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

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
    N = 2
    uncorrelated_results = []
    for n in range(2, 80, 4):
        counts[:] = orig_counts
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, n)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        print("Medium grained uncorrelated: {} seconds".format(t.interval))
        uncorrelated_results.append(t.interval)

    counts[:] = orig_counts
    with Timer() as t:
        move_data_medium_grained(counts, src, dest, 100, orig_counts.size)
    assert counts.sum() == total, "Wrong total after move_data_medium_grained"
    print("Coarse grained uncorrelated: {} seconds".format(t.interval))
        

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
    N = 2
    correlated_results = []
    for n in range(2, 80, 4):
        counts[:] = orig_counts
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, n)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        print("Medium grained correlated: {} seconds".format(t.interval))
        correlated_results.append(t.interval)

    counts[:] = orig_counts
    with Timer() as t:
        move_data_medium_grained(counts, src, dest, 100, orig_counts.size)
    assert counts.sum() == total, "Wrong total after move_data_medium_grained"
    print("Coarse grained correlated: {} seconds".format(t.interval))

    plt.figure()
    plt.plot(range(2, 80, 4), uncorrelated_results, color='red', label="Random data exchange times")
    plt.plot(range(2, 80, 4), correlated_results, color='blue', label="Correlated data exchange times")
    plt.xlabel("N")
    plt.ylabel("Time")
    plt.legend()
    plt.show()
