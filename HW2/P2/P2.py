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
    N = range(1,33)
    T=[]
    for n in N:
        print "n is ", n
        counts[:] = orig_counts
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, n)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        print("Medium grained uncorrelated: {} seconds".format(t.interval))
        T.append(t.interval)

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
    N2 = range(1,33)
    T2=[]
    for n in N2:
        print "n is ", n
        counts[:] = orig_counts
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, n)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        print("Medium grained correlated: {} seconds".format(t.interval))
        T2.append(t.interval)
    
    plt.plot(N,T,label="Medium grained uncorrelated time")
    plt.plot(N2,T2,label="Medium grained correlated time")
    plt.xlabel("N")
    plt.ylabel("Time (s)")
    plt.title("P2 Time Comparison")
    plt.legend()
    plt.savefig("P2_fig")
