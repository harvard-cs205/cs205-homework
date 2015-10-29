# Note: includes all my comments from reviewing the skeleton code

import sys
sys.path.append('../util')

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import numpy as np
from timer import Timer
from parallel_vector import move_data_serial, move_data_fine_grained, move_data_medium_grained

import json

if __name__ == '__main__':

    ########################################
    # Generate some test data, first, uncorrelated
    ########################################

    orig_counts = np.arange(1000, dtype=np.int32) # Ordered array with 1k values between 0 and 999
    src = np.random.randint(1000, size=1000000).astype(np.int32) # Array with 1 mil. values between 0 and 999 (values correspond to indices in counts array)
    dest = np.random.randint(1000, size=1000000).astype(np.int32) # Array with 1 mil. values between 0 and 999 (values correspond to indices in counts array)

    total = orig_counts.sum() # = 499500

    # Initialize parameters for testing
    test_N = range(1, 101)
    test_uncor = dict()
    test_cor = dict()

    # serial move
    counts = orig_counts.copy() # Create copy of original counts array
    with Timer() as t:
        move_data_serial(counts, src, dest, 100) # 100 = number of iterations of outer loop
    assert counts.sum() == total, "Wrong total after move_data_serial" # Check that the sum hasn't changed
    runtime_serial_uncor = t.interval
    print("Serial uncorrelated: {} seconds".format(runtime_serial_uncor)) # Print runtime
    test_uncor['SERIAL'] =  runtime_serial_uncor # Store runtime for graph
    serial_counts = counts.copy()

    # fine grained
    counts[:] = orig_counts # Create copy of original counts array
    with Timer() as t:
        move_data_fine_grained(counts, src, dest, 100) # 100 = number of iterations of outer loop
    assert counts.sum() == total, "Wrong total after move_data_fine_grained"
    runtime_fine_uncor = t.interval
    print("Fine grained uncorrelated: {} seconds".format(runtime_fine_uncor))
    test_uncor['FINE'] =  runtime_fine_uncor # Store runtime for graph

    ########################################
    # You should explore different values for the number of locks in the medium
    # grained locking
    ########################################

    for N in test_N:
        counts[:] = orig_counts # Create copy of original counts array
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, N) # 100 = number of iterations of outer loop
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        runtime_coarse_uncor = t.interval
        print("Medium grained uncorrelated: {} seconds".format(runtime_coarse_uncor))
        test_uncor[N] = runtime_coarse_uncor # Store runtime for graph

    # Output results for graph
    fd = open("P2_results_uncor.json","w")
    json.dump(test_uncor, fd)
    fd.close()
    del test_uncor

    ########################################
    # Now use correlated data movement
    ########################################

    dest = src + np.random.randint(-10, 11, size=src.size) # All values (i.e. counts indices) are at most +/- 10 away
    dest[dest < 0] += 1000 # Ensure that no indices < 0 (out of bounds)
    dest[dest >= 1000] -= 1000 # Ensure that no indices >= 1000 (out of bounds)
    dest = dest.astype(np.int32)

    # serial move
    counts[:] = orig_counts # Create copy of original counts array
    with Timer() as t:
        move_data_serial(counts, src, dest, 100) # 100 = number of iterations of outer loop
    assert counts.sum() == total, "Wrong total after move_data_serial"
    runtime_serial_cor = t.interval
    print("Serial correlated: {} seconds".format(runtime_serial_cor))
    test_cor['SERIAL'] =  runtime_serial_cor # Store runtime for graph
    serial_counts = counts.copy()

    # fine grained
    counts[:] = orig_counts # Create copy of original counts array
    with Timer() as t:
        move_data_fine_grained(counts, src, dest, 100) # 100 = number of iterations of outer loop
    assert counts.sum() == total, "Wrong total after move_data_fine_grained"
    runtime_fine_cor = t.interval
    print("Fine grained correlated: {} seconds".format(runtime_fine_cor))
    test_cor['FINE'] =  runtime_fine_cor # Store runtime for graph

    ########################################
    # You should explore different values for the number of locks in the medium
    # grained locking
    ########################################

    for N in test_N:
        counts[:] = orig_counts # Create copy of original counts array
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, N) # 100 = number of iterations of outer loop
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        runtime_coarse_cor = t.interval
        print("Medium grained correlated: {} seconds".format(runtime_coarse_cor))
        test_cor[N] = runtime_coarse_cor # Store runtime for graph
    
    # Output results for graph
    fd = open("P2_results_cor.json","w")
    json.dump(test_cor, fd)
    fd.close()
    del test_cor
