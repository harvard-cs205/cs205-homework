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


    serial_unc, serial_cor = [], []
    med_unc, med_cor = [], []
    fine_unc, fine_cor = [], []

    for N in [1, 5, 10, 25, 50, 100]:
        orig_counts = np.arange(1000, dtype=np.int32)
        src = np.random.randint(1000, size=1000000).astype(np.int32)
        dest = np.random.randint(1000, size=1000000).astype(np.int32)

        total = orig_counts.sum()
        print 'N =', N
        # serial move
        counts = orig_counts.copy()
        with Timer() as t:
            move_data_serial(counts, src, dest, 100)
        assert counts.sum() == total, "Wrong total after move_data_serial"
        print("Serial uncorrelated: {} seconds".format(t.interval))
        t_sunc = t.interval
        serial_counts = counts.copy()


        # fine grained
        counts[:] = orig_counts
        with Timer() as t:
            move_data_fine_grained(counts, src, dest, 100)
        assert counts.sum() == total, "Wrong total after move_data_fine_grained"
        print("Fine grained uncorrelated: {} seconds".format(t.interval))
        t_func = t.interval
        ########################################
        # You should explore different values for the number of locks in the medium
        # grained locking
        ########################################
        # N = 10



        counts[:] = orig_counts
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, N)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        print("Medium grained uncorrelated: {} seconds".format(t.interval))
        t_munc = t.interval
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
        t_scor = t.interval

        serial_counts = counts.copy()

        # fine grained
        counts[:] = orig_counts
        with Timer() as t:
            move_data_fine_grained(counts, src, dest, 100)
        assert counts.sum() == total, "Wrong total after move_data_fine_grained"
        print("Fine grained correlated: {} seconds".format(t.interval))
        t_fcor = t.interval

        ########################################
        # You should explore different values for the number of locks in the medium
        # grained locking
        ########################################
        # N = 10
        counts[:] = orig_counts
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, N)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        print("Medium grained correlated: {} seconds".format(t.interval))
        t_mcor = t.interval

        serial_unc.append(t_sunc)
        serial_cor.append(t_scor)
        med_unc.append(t_munc)
        med_cor.append(t_mcor)
        fine_unc.append(t_func)
        fine_cor.append(t_fcor)

    print serial_unc, serial_cor
    print med_unc, med_cor
    print fine_unc, fine_cor