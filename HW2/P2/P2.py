
# coding: utf-8

# In[1]:

# %load P2.py
import sys
sys.path.append('../util')

import set_compiler
set_compiler.install()

import pyximport
pyximport.install(reload_support=True)

import numpy as np
from timer import Timer
from parallel_vector import move_data_serial, move_data_fine_grained, move_data_medium_grained


# In[150]:

import parallel_vector
import matplotlib.pyplot as plt
import seaborn


# In[130]:

# if __name__ == '__main__':
########################################
# Generate some test data, first, uncorrelated
########################################
count_num = 1000
orig_counts = np.arange(count_num, dtype=np.int32)
src = np.random.randint(count_num, size=count_num**2).astype(np.int32)
dest = np.random.randint(count_num, size=count_num**2).astype(np.int32)

total = orig_counts.sum()

# serial move
counts = orig_counts.copy()


# In[131]:

with Timer() as t_serial:
    move_data_serial(counts, src, dest, 100)
assert counts.sum() == total, "Wrong total after move_data_serial"
print("Serial uncorrelated: {} seconds".format(t_serial.interval))
serial_counts = counts.copy()


# In[132]:

# fine grained
counts[:] = orig_counts
with Timer() as t_fine:
    parallel_vector.move_data_fine_grained(counts, src, dest, 100)
assert counts.sum() == total, "Wrong total after move_data_fine_grained"
print("Fine grained uncorrelated: {} seconds".format(t_fine.interval))


# In[134]:

########################################
# Explore different values for the number of locks
########################################
N_list = np.logspace(0, 3, 20)
t_medium = []
for N in N_list:
    counts[:] = orig_counts
    with Timer() as t:
        parallel_vector.move_data_medium_grained(counts, src, dest, 100, N)
    assert counts.sum() == total, "Wrong total after move_data_medium_grained"
    t_medium.append(t.interval)
#     print("Medium grained uncorrelated: {} seconds".format(t.interval))


# In[138]:

########################################
# Now use correlated data movement
########################################
dest = src + np.random.randint(-10, 11, size=src.size)
dest[dest < 0] += 1000
dest[dest >= 1000] -= 1000
dest = dest.astype(np.int32)

# serial move
counts[:] = orig_counts
with Timer() as t_corr_serial:
    move_data_serial(counts, src, dest, 100)
assert counts.sum() == total, "Wrong total after move_data_serial"
print("Serial correlated: {} seconds".format(t_corr_serial.interval))
serial_counts = counts.copy()


# In[139]:

# fine grained
counts[:] = orig_counts
with Timer() as t_corr_fine:
    parallel_vector.move_data_fine_grained(counts, src, dest, 100)
assert counts.sum() == total, "Wrong total after move_data_fine_grained"
print("Fine grained correlated: {} seconds".format(t_corr_fine.interval))


# In[140]:

########################################
# You should explore different values for the number of locks in the medium
# grained locking
########################################
N_list = np.logspace(0, 3, 20)
t_corr_medium = []
for N in N_list:
    counts[:] = orig_counts
    with Timer() as t:
        parallel_vector.move_data_medium_grained(counts, src, dest, 100, N)
    assert counts.sum() == total, "Wrong total after move_data_medium_grained"
    t_corr_medium.append(t.interval)
# print("Medium grained correlated: {} seconds".format(t.interval))


# In[158]:

plt.figure()

# Plot correlated transfer results
plt.plot(1, t_corr_fine.interval / 100, 'ro')
plt.plot(N_list, [x / 100 for x in t_corr_medium], 'k-')
plt.plot([1, 1e3], 2 * [t_corr_serial.interval / 100], 'g-',
        linewidth = 3)
plt.xscale('log')
plt.xlim([10**-0.2, 1500])
plt.xlabel('Size of lock')
plt.ylabel('Avg time to complete a million moves (sec)')
plt.legend(['Fine-grained', 
            'Medium-grained (with different lock sizes)', 
            'Serial'], loc='lower center')

# Plot uncorrelated transfer results
plt.plot(1, t_fine.interval / 100, 'rs')
plt.plot(N_list, [x / 100 for x in t_medium], 'k--')
plt.plot([1, 1e3], 2 * [t_serial.interval / 100], 'g:',
        linewidth = 3)
plt.xscale('log')
plt.xlim([10**-0.2, 1500])
plt.xlabel('Log size of lock')
plt.ylabel('Avg time to complete a million moves (sec)')
plt.legend(['Fine-grained (correlated)', 
            'Medium-grained (correlated)', 
            'Serial (correlated)',
            'Fine-grained (uncorrelated)', 
            'Medium-grained (uncorrelated)', 
            'Serial (uncorrelated)'], loc='upper left')
plt.title('Comparison of different grain sizes\n on correlated and uncorrelated memory transfer')
# plt.show()
plt.savefig('P2_transfer_times.png')

