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

    t_serial = [0]*3
    t_fine = [0]*3
    t_medium = np.array([0.0]*21)
    t_medium = t_medium.reshape(3,7)

    for i in range(3):
        # serial move
        counts = orig_counts.copy()
        with Timer() as t:
            move_data_serial(counts, src, dest, 100)
        assert counts.sum() == total, "Wrong total after move_data_serial"
        print("Serial uncorrelated: {} seconds".format(t.interval))
        serial_counts = counts.copy()
        t_serial[i] = t.interval

        # fine grained
        counts[:] = orig_counts
        with Timer() as t:
            move_data_fine_grained(counts, src, dest, 100)
        assert counts.sum() == total, "Wrong total after move_data_fine_grained"
        print("Fine grained uncorrelated: {} seconds".format(t.interval))
        t_fine[i] = t.interval

        ########################################
        # You should explore different values for the number of locks in the medium
        # grained locking
        ########################################
        for N in [j*2 for j in range(1,8)]:
            counts[:] = orig_counts
            with Timer() as t:
                move_data_medium_grained(counts, src, dest, 100, N)
            assert counts.sum() == total, "Wrong total after move_data_medium_grained"
            print("Medium grained uncorrelated: {} seconds".format(t.interval))
            t_medium[i, N/2-1] = t.interval

    ########################################
    # Now use correlated data movement
    ########################################
    dest = src + np.random.randint(-10, 11, size=src.size)
    dest[dest < 0] += 1000
    dest[dest >= 1000] -= 1000
    dest = dest.astype(np.int32)

    t_serial_c = [0]*3
    t_fine_c = [0]*3
    t_medium_c = np.array([0.0]*21)
    t_medium_c = t_medium_c.reshape(3,7)

    for i in range(3):
        # serial move
        counts[:] = orig_counts
        with Timer() as t:
            move_data_serial(counts, src, dest, 100)
        assert counts.sum() == total, "Wrong total after move_data_serial"
        print("Serial correlated: {} seconds".format(t.interval))
        serial_counts = counts.copy()
        t_serial_c[i] = t.interval

        # fine grained
        counts[:] = orig_counts
        with Timer() as t:
            move_data_fine_grained(counts, src, dest, 100)
        assert counts.sum() == total, "Wrong total after move_data_fine_grained"
        print("Fine grained correlated: {} seconds".format(t.interval))
        t_fine_c[i] = t.interval

        ########################################
        # You should explore different values for the number of locks in the medium
        # grained locking
        ########################################
        for N in [j*2 for j in range(1,8)]:
            counts[:] = orig_counts
            with Timer() as t:
                move_data_medium_grained(counts, src, dest, 100, N)
            assert counts.sum() == total, "Wrong total after move_data_medium_grained"
            print("Medium grained correlated: {} seconds".format(t.interval))
            t_medium_c[i, N/2-1] = t.interval

print t_serial
print t_serial_c
print t_fine
print t_fine_c
print t_medium
print t_medium_c

t_medium_trans = [(t_medium[0,i]+t_medium[1,i]+t_medium[2,i])/3 for i in range(7)]
t_medium_c_trans = [(t_medium_c[0,i]+t_medium_c[1,i]+t_medium_c[2,i])/3 for i in range(7)]

plt.hlines(np.average(t_serial), 2, 14, label='serial', color='r')
plt.hlines(np.average(t_serial_c), 2, 14, label='serial, corr', color='k')
plt.hlines(np.average(t_fine), 2, 14, label='fine', color='y')
plt.hlines(np.average(t_fine_c), 2, 14, label='fine, corr', color='orange')
plt.plot([i*2 for i in range(1,8)], t_medium_trans, label='medium')
plt.plot([i*2 for i in range(1,8)], t_medium_c_trans, label='medium, corr')
plt.legend()
plt.ylim(0, 15)
plt.xlabel('N')
plt.ylabel('t')
plt.show()