import sys
sys.path.append('../util')

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import numpy as np
import seaborn as sns
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
    N = 10
    counts[:] = orig_counts
    with Timer() as t:
        move_data_medium_grained(counts, src, dest, 100, N)
    assert counts.sum() == total, "Wrong total after move_data_medium_grained"
    print("Medium grained uncorrelated: {} seconds".format(t.interval))

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
    N = 10
    counts[:] = orig_counts
    with Timer() as t:
        move_data_medium_grained(counts, src, dest, 100, N)
    assert counts.sum() == total, "Wrong total after move_data_medium_grained"
    print("Medium grained correlated: {} seconds".format(t.interval))

    ########################################
    # EXPERIMENT WITH N
    # medium grained locking with graphs
    ########################################

    orig_counts = np.arange(1000, dtype=np.int32)
    src = np.random.randint(1000, size=1000000).astype(np.int32)
    dest = np.random.randint(1000, size=1000000).astype(np.int32)
    total = orig_counts.sum()
    counts = orig_counts.copy()

    Ns = [2**i for i in xrange(0, 6)]
    uncorrelated_times = []
    correlated_times = []

    for N in Ns:
        counts[:] = orig_counts
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, N)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        print("Medium grained uncorrelated: {} seconds".format(t.interval))
        uncorrelated_times += [t.interval]

    dest = src + np.random.randint(-10, 11, size=src.size)
    dest[dest < 0] += 1000
    dest[dest >= 1000] -= 1000
    dest = dest.astype(np.int32)

    for N in Ns:
        counts[:] = orig_counts
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, N)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        print("Medium grained correlated: {} seconds".format(t.interval))
        correlated_times += [t.interval]

    sns.set_style("darkgrid")
    f, ax = plt.subplots(1)
    plt.plot(Ns, uncorrelated_times, label="Uncorrelated")
    plt.plot(Ns, correlated_times, label="Correlated")
    plt.xlabel('N')
    plt.ylabel('Time (sec.)')
    plt.title('Execution time for various values of N')
    ax.legend(loc='lower right')
    ax.set_ylim(0,)
    plt.show()
