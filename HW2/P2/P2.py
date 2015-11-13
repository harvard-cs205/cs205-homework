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

    reps = 3
    N = range(5,101,5) + [1000]
    times = []
    
    # serial move
    counts = orig_counts.copy()    
    for r in range(reps):
        counts[:] = orig_counts
        with Timer() as t:
            move_data_serial(counts, src, dest, 100)
        assert counts.sum() == total, "Wrong total after move_data_serial"
        times.append(t.interval)
    print("Serial uncorrelated: {} seconds".format(min(times)))
    serial_counts = counts.copy()

    # fine grained
    times = []
    for r in range(reps):
        counts[:] = orig_counts
        with Timer() as t:
            move_data_fine_grained(counts, src, dest, 100)
        assert counts.sum() == total, "Wrong total after move_data_fine_grained"
        times.append(t.interval)
    print("Fine grained uncorrelated: {} seconds".format(min(times)))
        
    # middle grained
    mid_unc = []
    for n in N:
        times = []
        for r in range(reps):
            counts[:] = orig_counts
            with Timer() as t:
                move_data_medium_grained(counts, src, dest, 100, n)
            assert counts.sum() == total, "Wrong total after move_data_medium_grained"
            times.append(t.interval)
        print("Medium grained uncorrelated at {}: {} seconds".format(n,min(times)))
        mid_unc.append(min(times))
        
    ########################################
    # Now use correlated data movement
    ########################################
    dest = src + np.random.randint(-10, 11, size=src.size)
    dest[dest < 0] += 1000
    dest[dest >= 1000] -= 1000
    dest = dest.astype(np.int32)

    # serial move
    times = []
    for r in range(reps):
        counts[:] = orig_counts
        with Timer() as t:
            move_data_serial(counts, src, dest, 100)
        assert counts.sum() == total, "Wrong total after move_data_serial"
        times.append(t.interval)
    print("Serial correlated: {} seconds".format(min(times)))
    serial_counts = counts.copy()

    # fine grained
    times = []
    for r in range(reps):
        counts[:] = orig_counts
        with Timer() as t:
            move_data_fine_grained(counts, src, dest, 100)
        assert counts.sum() == total, "Wrong total after move_data_fine_grained"
        times.append(t.interval)
    print("Fine grained correlated: {} seconds".format(min(times)))

    # middle grained
    mid_cor = []
    for n in N:
        times = []
        for r in range(reps):
            counts[:] = orig_counts
            with Timer() as t:
                move_data_medium_grained(counts, src, dest, 100, n)
            assert counts.sum() == total, "Wrong total after move_data_medium_grained"
            times.append(t.interval)
        print("Medium grained correlated at {}: {} seconds".format(n, min(times)))
        mid_cor.append(min(times))

    plt.figure()
    plt.plot(N[:-2], mid_unc[:-2],color='red')
    plt.plot(N[:-2], mid_cor[:-2],color='black')
    plt.legend(['Uncorrelated','Correlated'],loc='upper right')
    plt.xlabel("Graining (N)")
    plt.ylabel("Time of exchanging data (minimum over three reps)")
    plt.show()
