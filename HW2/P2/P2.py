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
    """
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
    """
    uncor_times = []
    cor_times = []
    N_vals = np.array([1,2,5,10,20,21,50,100,200,500,1000])
    for N in N_vals:

        ########################################
        # Generate some test data, first, uncorrelated
        ########################################
        orig_counts = np.arange(1000, dtype=np.int32)
        src = np.random.randint(1000, size=1000000).astype(np.int32)
        dest = np.random.randint(1000, size=1000000).astype(np.int32)

        total = orig_counts.sum()
        counts = orig_counts.copy()
        counts[:] = orig_counts
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, N)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        #print("Medium grained uncorrelated: {} seconds".format(t.interval))
        print("Uncorrelated, N = " + str(N) + ": " + str(t.interval) + " seconds")
        uncor_times.append(t.interval)
    
        ########################################
        # Now use correlated data movement
        ########################################
        dest = src + np.random.randint(-10, 11, size=src.size)
        dest[dest < 0] += 1000
        dest[dest >= 1000] -= 1000
        dest = dest.astype(np.int32)

        counts[:] = orig_counts
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, N)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        print("  Correlated, N = " + str(N) + ": " + str(t.interval) + " seconds")
        #print("Medium grained correlated: {} seconds".format(t.interval))
        cor_times.append(t.interval)
    
    uncor_times = np.array(uncor_times)
    print uncor_times
    cor_times = np.array(cor_times)
    print cor_times

    plt.figure()
    plt.plot(N_vals, uncor_times, lw=3, color = "red", label = "Uncorrelated")    
    plt.plot(N_vals,   cor_times, lw=3, color= "blue", label =   "Correlated")
    plt.xscale('log')
    plt.xlabel("N")
    plt.ylabel("Time (seconds)")
    plt.title("Comparison of Medium-grained Runtimes with varying N")
    plt.legend(loc='best')
    plt.show() 
