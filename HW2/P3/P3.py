
# coding: utf-8

# In[1]:

import sys
import os.path
sys.path.append(os.path.join('..', 'util'))

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import mandelbrot
from timer import Timer

import pylab as plt
import numpy as np
#get_ipython().magic(u'matplotlib inline')

# create coordinates, along with output count array
def make_coords(center=(-0.575 - 0.575j),
                width=0.0025,
                count=4000):

    x = np.linspace(start=(-width / 2), stop=(width / 2), num=count)
    xx = center + (x + 1j * x[:, np.newaxis]).astype(np.complex64)
    return xx, np.zeros_like(xx, dtype=np.uint32)


# In[ ]:

in_coords, out_counts = make_coords()


# In[ ]:

with Timer() as t:
        mandelbrot.avx_mandelbrot(in_coords[0:2,24:80], out_counts[0:2,24:80], 1, 100)
seconds = t.interval

print("{} Million Complex FMAs in {} seconds, {} million Complex FMAs / second".format(out_counts.sum() / 1e6, seconds, (out_counts.sum() / seconds) / 1e6))

plt.imshow(np.log(out_counts))
plt.show()

