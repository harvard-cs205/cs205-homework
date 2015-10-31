
# coding: utf-8

# In[1]:

import sys
sys.path.append('../util')

import set_compiler
set_compiler.install()

import pyximport
pyximport.install(reload_support=True)

import numpy as np
from timer import Timer
import mandelbrot


# In[ ]:

from matplotlib import pyplot as plt


# In[2]:

# create coordinates, along with output count array
def make_coords(center=(-0.575 - 0.575j),
                width=0.0025,
                count=4000):

    x = np.linspace(start=(-width / 2), stop=(width / 2), num=count)
    xx = center + (x + 1j * x[:, np.newaxis]).astype(np.complex64)
    return xx, np.zeros_like(xx, dtype=np.uint32)


# In[188]:

def make_subplots(n_threads_i, tit):
    plt.subplot(2, 3, n_threads_i + 1)
    plt.imshow(out_counts)
    plt.xticks([])
    plt.yticks([])
    plt.title(tit)


# In[193]:

n_threads_list = [1, 2, 4]
# Without AVX
threads_time = []
for n_threads_i, n_threads in enumerate(n_threads_list):
    in_coords, out_counts = make_coords()
    with Timer() as t:
        out_counts = mandelbrot.mandelbrot(in_coords, 
                                           out_counts, 
                                           n_threads)
    threads_time.append(t.interval)
    make_subplots(n_threads_i, 'No AVX, {}threads'.format(n_threads))

# With AVX
in_coords, out_counts_avx = make_coords()
threads_time_avx = []
for n_threads_i, n_threads in enumerate(n_threads_list):
    with Timer() as t:
        out_counts_avx = mandelbrot.mandelbrot_avx(in_coords, 
                                               out_counts_avx, 
                                               n_threads)
    threads_time_avx.append(t.interval)
    make_subplots(n_threads_i + 3, 'With AVX, {}threads'.format(n_threads))
    
plt.savefig('P3_mandelbrot_correctness.png')


# In[200]:

plt.plot(n_threads_list, threads_time, 'ko-')
plt.plot(n_threads_list, threads_time_avx, 'bo-')
plt.xlabel('Number of threads')
plt.ylabel('Time taken (sec)')
plt.xticks([1, 2, 4])
plt.title('Mandelbrot: improvement from parallelization')
plt.savefig('P3_mandelbrot_speedimprovement.png')

