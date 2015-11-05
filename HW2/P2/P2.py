import sys
sys.path.append('../util')

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
    times = []
    for i in range(1):
        counts = orig_counts.copy()
        with Timer() as t:
            move_data_serial(counts, src, dest, 100)
        assert counts.sum() == total, "Wrong total after move_data_serial"
        
        current_time = t.interval
        print("Serial uncorrelated: {} seconds".format(current_time))
        times.append(current_time)

    print "Average", np.mean(times), "\n\n"

    serial_counts = counts.copy()

    # fine grained
    times = []
    for i in range(1):
        counts[:] = orig_counts
        with Timer() as t:
            move_data_fine_grained(counts, src, dest, 100)
        assert counts.sum() == total, "Wrong total after move_data_fine_grained"

        current_time = t.interval
        print("Fine grained uncorrelated: {} seconds".format(current_time))
        times.append(current_time)

    print "Average", np.mean(times), "\n\n"

    ########################################
    # You should explore different values for the number of locks in the medium
    # grained locking
    ########################################
    N = 10
    times = []
    for N in [1,5,10,20,50,100]:
        counts[:] = orig_counts
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, N)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"

        current_time = t.interval
        print("Medium grained uncorrelated with {} locks: {} seconds".format(N, current_time))
        times.append(current_time)

    #print "Average", np.mean(times), "\n\n"

    ########################################
    # Now use correlated data movement
    ########################################
    dest = src + np.random.randint(-10, 11, size=src.size)
    dest[dest < 0] += 1000
    dest[dest >= 1000] -= 1000
    dest = dest.astype(np.int32)

    # serial move
    times = []
    for i in range(1):
        counts[:] = orig_counts
        with Timer() as t:
            move_data_serial(counts, src, dest, 100)
        assert counts.sum() == total, "Wrong total after move_data_serial"

        current_time = t.interval
        print("Serial correlated: {} seconds".format(current_time))
        times.append(current_time)

    print "Average", np.mean(times), "\n\n"

    serial_counts = counts.copy()

    # fine grained
    times = []
    for i in range(1):
        counts[:] = orig_counts
        with Timer() as t:
            move_data_fine_grained(counts, src, dest, 100)
        assert counts.sum() == total, "Wrong total after move_data_fine_grained"

        current_time = t.interval
        print("Fine grained correlated: {} seconds".format(current_time))
        times.append(current_time)

    print "Average", np.mean(times), "\n\n"

    ########################################
    # You should explore different values for the number of locks in the medium
    # grained locking
    ########################################
    N = 10
    times = []
    for N in [1,5,10,20,50,100]:
        counts[:] = orig_counts
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, N)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"

        current_time = t.interval
        print("Medium grained correlated with {} locks: {} seconds".format(N, current_time))
        times.append(current_time)

    #print "Average", np.mean(times), "\n\n"
