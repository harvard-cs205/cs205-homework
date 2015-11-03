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


number_lock = 20
output = np.zeros((number_lock,2))
for i in range(number_lock):
	threads = 4
	N = i
########################################
# loop through and generate graphs for different number of threads
	orig_counts = np.arange(1000, dtype=np.int32)
	src = np.random.randint(1000, size=1000000).astype(np.int32)
	dest = np.random.randint(1000, size=1000000).astype(np.int32)

	total = orig_counts.sum()

    # serial uncoorelated
	counts = orig_counts.copy()
    ########################################
    # You should explore different values for the number of locks in the medium
    # grained locking
    ########################################
	N = 10
	counts[:] = orig_counts
	with Timer() as t_med_uncor:
		move_data_medium_grained(counts, src, dest, 100, N, threads)
	assert counts.sum() == total, "Wrong total after move_data_medium_grained"
	output[i,0] = t_med_uncor.interval

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
	N = 10
	counts[:] = orig_counts
	with Timer() as t_med_cor:
		move_data_medium_grained(counts, src, dest, 100, N, threads)
	assert counts.sum() == total, "Wrong total after move_data_medium_grained"
	output[i,1] = t_med_cor.interval


names = ['Medium Uncorelated', 'Medium Corelated']
color = ['r', 'b']
plt.figure(figsize=(12,12))
for i in range(2):
	x = range(number_lock)
	y = output[:,i]
	plt.plot(x, y, color[i], label = names[i])
	plt.title('Lock Performance')
	plt.xlabel('Number of Locks')
	plt.ylabel('Time')
	plt.legend(loc = 'lower right')
plt.savefig('P2_plot_locks.png')


