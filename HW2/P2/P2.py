import sys
sys.path.append('../util')

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import numpy as np
from timer import Timer
from parallel_vector import move_data_serial, move_data_fine_grained, move_data_medium_grained, time_locks_fine_grained, time_locks_medium_grained
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ########################################
    # Generate some test data, first, uncorrelated
    ########################################
    orig_counts = np.arange(1000, dtype=np.int32)
    src = np.random.randint(1000, size=1000000).astype(np.int32)
    dest = np.random.randint(1000, size=1000000).astype(np.int32)

    total = orig_counts.sum()
    uncorrelated_lock_times = []
    uncorrelated_times = []
    correlated_lock_times = []
    correlated_times = []

    # serial move
    counts = orig_counts.copy()
    with Timer() as t:
        move_data_serial(counts, src, dest, 100)
    assert counts.sum() == total, "Wrong total after move_data_serial"
    print("Serial uncorrelated: {} seconds".format(t.interval))
    serial_counts = counts.copy()
    
    # lock timing
    with Timer() as t:
        time_locks_fine_grained(orig_counts, src, dest, 100)
    print ("Time for locks fine uncorrelated: {} seconds".format(t.interval))
    uncorrelated_lock_times.append(t.interval)

    # fine grained
    counts[:] = orig_counts
    with Timer() as t:
        move_data_fine_grained(counts, src, dest, 100)
    assert counts.sum() == total, "Wrong total after move_data_fine_grained"
    print("Fine grained uncorrelated: {} seconds".format(t.interval))
    uncorrelated_times.append(t.interval)    
    
    ########################################
    # You should explore different values for the number of locks in the medium
    # grained locking
    ########################################
    N_arr = [5,10,20,50,100]
    counts[:] = orig_counts
    for N in N_arr:
        #lock timing
        with Timer() as t:
            time_locks_medium_grained(orig_counts, src, dest, 100, N)
        print ("Time for locks medium uncorrelated: N={},  {} seconds".format(N, t.interval))
        uncorrelated_lock_times.append(t.interval)        
        
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, N)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        print("Medium uncorrelated: N={}, {} seconds".format(N, t.interval))
        uncorrelated_times.append(t.interval)
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
    
    # lock timing
    with Timer() as t:
        time_locks_fine_grained(orig_counts, src, dest, 100)
    print ("Time for locks fine correlated: {} seconds".format(t.interval))
    correlated_lock_times.append(t.interval)

    # fine grained
    counts[:] = orig_counts
    with Timer() as t:
        move_data_fine_grained(counts, src, dest, 100)
    assert counts.sum() == total, "Wrong total after move_data_fine_grained"
    print("Fine grained correlated: {} seconds".format(t.interval))
    correlated_times.append(t.interval)

    ########################################
    # You should explore different values for the number of locks in the medium
    # grained locking
    ########################################
    counts[:] = orig_counts
    for N in N_arr:
        #lock timing
        with Timer() as t:
            time_locks_medium_grained(orig_counts, src, dest, 100, N)
        print ("Time for locks medium correlated: N={},  {} seconds".format(N, t.interval))
        correlated_lock_times.append(t.interval)
        
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, N)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        print("Medium correlated: N={}, {} seconds".format(N, t.interval))
        correlated_times.append(t.interval)
        
    N_arr = [1] + N_arr
    plt.plot(N_arr, uncorrelated_lock_times)
    plt.plot(N_arr, uncorrelated_times)
    plt.plot(N_arr, correlated_lock_times)
    plt.plot(N_arr, correlated_times)
    plt.legend(['Uncorr lock times', 'Uncorr times', 'Corr lock times', 'Corr times'], loc='upper left')
    plt.show()