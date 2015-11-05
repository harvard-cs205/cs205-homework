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
    N = 10
    counts[:] = orig_counts
    with Timer() as t:
        move_data_medium_grained(counts, src, dest, 100, N)
    assert counts.sum() == total, "Wrong total after move_data_medium_grained"
    print("Medium grained uncorrelated: {} seconds".format(t.interval))
    medUncorr_n10time = t.interval
    ##########################################
    # Exploring different values for N in medium grained locking uncorrelated
    ##########################################
    
    Ns = range(2,32,2) #[nn**2 for nn in range(1,11)]
    Ns.insert(0,1)
    timesUncorrelated = []
    for nn in Ns:
#        print "\n"
#        print "N is {0}".format(nn)
        counts[:] = orig_counts
        with Timer() as t:
           move_data_medium_grained(counts, src, dest, 100, nn)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained..."
#        print("Medium grained uncorrelated: {} seconds".format(t.interval))
        timesUncorrelated.append(t.interval)
    print "\n"
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
    medCorr_n10time = t.interval
    ##########################################
    # Exploring different values for N in medium grained locking correlated
    ##########################################
    Ns = range(2,32,2) #[nn**2 for nn in range(1,11)]
    Ns.insert(0,1)
    timesCorrelated = []
    for nn in Ns:
#        print "\n"
#	        print "N is {0}".format(nn)
        counts[:] = orig_counts
        with Timer() as t:
           move_data_medium_grained(counts, src, dest, 100, nn)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained..."
#        print("Medium grained correlated: {} seconds".format(t.interval))
        timesCorrelated.append(t.interval)
    print "\n"
    
    plt.figure()
    plt.plot(Ns,timesUncorrelated,lw=3,color='black',label="Uncorrelated Medium Grained Locking")
    plt.plot(Ns,timesCorrelated,lw=3,color='blue',label="Correlated Medium Grained Locking")
    plt.plot(10,medUncorr_n10time,'k*',markersize=10,lw=2,label="Uncorrelated Baseline (N=10)")
    plt.plot(10,medCorr_n10time,'b*',markersize=10,lw=2,label="Correlated Baseline (N=10)")
    plt.xlabel("N")
    plt.ylabel("Time to Complete Moving of Data")
    plt.title("Comparison of Uncorrelated with Correlated Runtimes")
    plt.xlim([0,35])
    plt.ylim([0,15])
    plt.legend()
    plt.show()
    