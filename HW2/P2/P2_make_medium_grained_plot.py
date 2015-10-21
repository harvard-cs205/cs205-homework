import sys
sys.path.append('../util')

import set_compiler
set_compiler.install()

import pyximport

pyximport.install(reload_support=True)

# Annotate the file to make sure things are optimized
import subprocess
subprocess.call(["cython","-a","parallel_vector.pyx"])

import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('poster', font_scale=1.25)

import numpy as np
from timer import Timer
from parallel_vector import move_data_serial, move_data_fine_grained, move_data_medium_grained


# We want to make a plot based on changing N for both the correlated and
# uncorrelated
if __name__ == '__main__':
    ########################################
    # Generate some test data, first, uncorrelated
    ########################################
    orig_counts = np.arange(1000, dtype=np.int32)
    src = np.random.randint(1000, size=1000000).astype(np.int32)
    dest = np.random.randint(1000, size=1000000).astype(np.int32)

    total = orig_counts.sum()

    counts = orig_counts.copy()

    ########################################
    # You should explore different values for the number of locks in the medium
    # grained locking
    ########################################
    N_list = [1, 3, 5, 7, 9, 11, 15, 20]
    t_list = []
    for N in N_list:
        counts[:] = orig_counts
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, N)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        print("Medium grained uncorrelated: {} seconds".format(t.interval))
        t_list.append(t.interval)

    plt.plot(N_list, t_list, marker='.', ls='-')
    plt.xlabel('Number of Locks (N)')
    plt.ylabel('Time elapsed')
    plt.title('Uncorrelated Data')

    plt.savefig('uncorrelated_vs_N.png', dpi=100, bbox_inches='tight')
    plt.clf()

    ########################################
    # Now use correlated data movement
    ########################################
    dest = src + np.random.randint(-10, 11, size=src.size)
    dest[dest < 0] += 1000
    dest[dest >= 1000] -= 1000
    dest = dest.astype(np.int32)

    ########################################
    # You should explore different values for the number of locks in the medium
    # grained locking
    ########################################

    counts[:] = orig_counts
    t_list = []
    for N in N_list:
        with Timer() as t:
            move_data_medium_grained(counts, src, dest, 100, N)
        assert counts.sum() == total, "Wrong total after move_data_medium_grained"
        print("Medium grained correlated: {} seconds".format(t.interval))
        t_list.append(t.interval)

    plt.plot(N_list, t_list, marker='.', ls='-')
    plt.xlabel('Number of Locks (N)')
    plt.ylabel('Time elapsed')
    plt.title('Correlated Data')

    plt.savefig('correlated_vs_N.png', dpi=100, bbox_inches='tight')
    plt.clf()