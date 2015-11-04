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

    ### fine grained, 4 threads, uncorrelated, multilock ####
    # counts[:] = orig_counts
    # with Timer() as t:
    #     move_data_fine_grained(counts, src, dest, 100, 4)
    # assert counts.sum() == total, "Wrong total after move_data_fine_grained"
    # print("Fine grained uncorrelated: {} seconds".format(t.interval))
    # for t in threads:
        
    ### fine grained, multithread, uncorrelated ###
    # Save output for multiple Threads
    threads = range(1,10)
    time_fine_threads_uncor = []

    for th in threads:
        counts[:] = orig_counts
        with Timer() as t:
            move_data_fine_grained(counts, src, dest, 100, th)
        assert counts.sum() == total, "Wrong total after move_data_fine_grained"
        print("Number of Threads: {}".format(th))
        print("Fine grained uncorrelated: {} seconds".format(t.interval))
        time_fine_threads_uncor.append(t.interval)

    ########################################
    # You should explore different values for the number of locks in the medium
    # grained locking
    ########################################
    
    ### Medium grained, 4 threads, uncorrelated multi-lock ###
    # Create a range of N values for Part 1
    N_buffer = np.arange(1,21)

    # Create a few N values
    N_pts = [1,2,4,5,10]

    # Save Medium grain time results
    output_time_mg_uncor = []
    output_time_mg_uncor_pts = []
    
    # for N in N_buffer:
    #     counts[:] = orig_counts
    #     with Timer() as t:
    #         move_data_medium_grained(counts, src, dest, 100, N, 4)
    #     assert counts.sum() == total, "Wrong total after move_data_medium_grained"
    #     if (N == N_pts).any():
    #         print("Number of Locks: {}".format(N))
    #         print("Medium grained uncorrelated: {} seconds".format(t.interval))
    #         output_time_mg_uncor_pts.append(t.interval)
    #     output_time_mg_uncor.append(t.interval)

    ### Medium Grain, Multithread, Uncorrelated ###
    time_med_threads_uncor = []
    for th in threads:
        N=10
        counts[:] = orig_counts
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, N, th)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        print("Number of Threads: {}".format(th))
        print("Medium grained uncorrelated: {} seconds".format(t.interval))
        time_med_threads_uncor.append(t.interval)

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
    
    ### fine grained, 4 threads, correlated, multilock ####
    # counts[:] = orig_counts
    # with Timer() as t:
    #     move_data_fine_grained(counts, src, dest, 100, 4)
    # assert counts.sum() == total, "Wrong total after move_data_fine_grained"
    # print("Fine grained correlated: {} seconds".format(t.interval))

    ### fine grained, multithread, correlated ###
    time_fine_threads_cor = []
    for th in threads:
        counts[:] = orig_counts
        with Timer() as t:
            move_data_fine_grained(counts, src, dest, 100, th)
        assert counts.sum() == total, "Wrong total after move_data_fine_grained"
        print("Number of Locks: {}".format(th))
        print("Fine grained uncorrelated: {} seconds".format(t.interval))
        time_fine_threads_cor.append(t.interval)
    
    ########################################
    # You should explore different values for the number of locks in the medium
    # grained locking
    ########################################
    
    ### medium grained, 4 threads, correlated, multilock ####
    # output_time_mg_cor = []
    # output_time_mg_cor_pts = []
    # for N in N_buffer:
    #     counts[:] = orig_counts
    #     with Timer() as t:
    #         move_data_medium_grained(counts, src, dest, 100, N, 4)
    #     assert counts.sum() == total, "Wrong total after move_data_medium_grained"
    #     if (N == N_pts).any():
    #         print("Number of Locks: {}".format(N))
    #         print("Medium grained correlated: {} seconds".format(t.interval))
    #         output_time_mg_cor_pts.append(t.interval)
    #     output_time_mg_cor.append(t.interval)
    

    ### Medium Grain, Multithread, Correlated ###
    time_med_threads_cor = []
    N=10
    for th in threads:
        counts[:] = orig_counts
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, N, th)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        print("Number of Threads: {}".format(th))
        print("Medium grained uncorrelated: {} seconds".format(t.interval))
        time_med_threads_cor.append(t.interval)


    ### Multi-N ploting ####
    # plt.figure(figsize=(10,8))
    # plt.plot(N_buffer, output_time_mg_uncor)
    # plt.plot(N_buffer, output_time_mg_cor)
    # plt.scatter(N_pts, output_time_mg_uncor_pts, s=50, c='Red', label=u'UnCorrelated')
    # plt.scatter(N_pts, output_time_mg_cor_pts, s=50, c='Green', label=u'Correlated')
    # plt.title("Time of Array Suffle Correlated vs. Uncorrelated (Medium Grain Locking)")
    # plt.xlabel("N")
    # plt.ylabel("Completation Time")
    # plt.legend(loc=2)
    # plt.show()

    ### Multithread ploting ####
    plt.figure(figsize=(10,8))
    
    # Uncorrelated 
    plt.plot(threads, time_fine_threads_uncor, label=u'Fine Uncorrelated')
    plt.plot(threads, time_med_threads_uncor, label=u'Medium Uncorrelated')
    
    # Correlated
    plt.plot(threads, time_fine_threads_cor, label=u'Fine Correlated')
    plt.plot(threads, time_med_threads_cor, label=u'Medium Correlated')
    
    # plt.scatter(N_pts, output_time_mg_uncor_pts, s=50, c='Red', label=u'UnCorrelated')
    # plt.scatter(N_pts, output_time_mg_cor_pts, s=50, c='Green', label=u'Correlated')
    plt.title("Time of Multithreaded Array Suffle Correlated vs. Uncorrelated")
    plt.xlabel("Threads")
    plt.ylabel("Completation Time")
    plt.legend(loc=2)
    plt.show()    
