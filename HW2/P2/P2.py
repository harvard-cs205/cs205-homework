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
    print "Original Total", total
    
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
    N = 1
    counts[:] = orig_counts
    with Timer() as t:
        move_data_medium_grained(counts, src, dest, 100, N)
    assert counts.sum() == total, "Wrong total after move_data_medium_grained"
    print("Medium grained 1 uncorrelated: {} seconds".format(t.interval))
    
    N = 2
    counts[:] = orig_counts
    with Timer() as t:
        move_data_medium_grained(counts, src, dest, 100, N)
    assert counts.sum() == total, "Wrong total after move_data_medium_grained"
    print("Medium grained 2 uncorrelated: {} seconds".format(t.interval))
    
    N = 3
    counts[:] = orig_counts
    with Timer() as t:
        move_data_medium_grained(counts, src, dest, 100, N)
    assert counts.sum() == total, "Wrong total after move_data_medium_grained"
    print("Medium grained 3 uncorrelated: {} seconds".format(t.interval))
    
    N = 4
    counts[:] = orig_counts
    with Timer() as t:
        move_data_medium_grained(counts, src, dest, 100, N)
    assert counts.sum() == total, "Wrong total after move_data_medium_grained"
    print("Medium grained 4 uncorrelated: {} seconds".format(t.interval))
    
    N = 5
    counts[:] = orig_counts
    with Timer() as t:
        move_data_medium_grained(counts, src, dest, 100, N)
    assert counts.sum() == total, "Wrong total after move_data_medium_grained"
    print("Medium grained 5 uncorrelated: {} seconds".format(t.interval))

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
    N = 30
    counts[:] = orig_counts
    with Timer() as t:
        move_data_medium_grained(counts, src, dest, 100, N)
    assert counts.sum() == total, "Wrong total after move_data_medium_grained"
    print("Medium grained 30 correlated: {} seconds".format(t.interval))
    
    N = 35
    counts[:] = orig_counts
    with Timer() as t:
        move_data_medium_grained(counts, src, dest, 100, N)
    assert counts.sum() == total, "Wrong total after move_data_medium_grained"
    print("Medium grained 35 correlated: {} seconds".format(t.interval))
    
    N = 40
    counts[:] = orig_counts
    with Timer() as t:
        move_data_medium_grained(counts, src, dest, 100, N)
    assert counts.sum() == total, "Wrong total after move_data_medium_grained"
    print("Medium grained 40 correlated: {} seconds".format(t.interval))
    
    N = 45
    counts[:] = orig_counts
    with Timer() as t:
        move_data_medium_grained(counts, src, dest, 100, N)
    assert counts.sum() == total, "Wrong total after move_data_medium_grained"
    print("Medium grained 45 correlated: {} seconds".format(t.interval))
    
    N = 50
    counts[:] = orig_counts
    with Timer() as t:
        move_data_medium_grained(counts, src, dest, 100, N)
    assert counts.sum() == total, "Wrong total after move_data_medium_grained"
    print("Medium grained 50 correlated: {} seconds".format(t.interval))

