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
    # N = 10
    # counts[:] = orig_counts
    # with Timer() as t:
    #     move_data_medium_grained(counts, src, dest, 100, N)
    # assert counts.sum() == total, "Wrong total after move_data_medium_grained"
    # print("Medium grained uncorrelated: {} seconds".format(t.interval))

    performance = []
    Ns = [1,2,3,4,5,6,7,8,9,10]
    for i in Ns:
        N = i 
        counts[:] = orig_counts
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, N)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        print "N", N
        print("Medium grained uncorrelated: {} seconds".format(t.interval))
        performance.append(t.interval)

    plt.scatter(Ns, performance)
    plt.xlabel('N')
    plt.ylabel('Time (seconds)')
    plt.title('Running time versus N value for correlated values')
    plt.savefig('performance_uncorrelated.png')


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
    
    performance = []
    Ns2 = [5,10,18,19,20, 21,22,23,24,25,50,100]
    
    for i in Ns2: #,18,19,20,21,22,23,24,50,100]
        N = i     
        counts[:] = orig_counts
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, N)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        print "N", N
        print("Medium grained correlated: {} seconds".format(t.interval))
        performance.append(t.interval)

    plt.scatter(Ns2, performance)
    plt.xlabel('N')
    plt.ylabel('Time (seconds)')
    plt.title('Running time versus N value for correlated values')
    plt.savefig('performance_correlated.png')
