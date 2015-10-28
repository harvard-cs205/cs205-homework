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


# create coordinates, along with output count array
def make_coords(center=(-0.575 - 0.575j),
                width=0.0025,
                count=4000):

    x = np.linspace(start=(-width / 2), stop=(width / 2), num=count)
    xx = center + (x + 1j * x[:, np.newaxis]).astype(np.complex64)
    return xx, np.zeros_like(xx, dtype=np.uint32)


if __name__ == '__main__':
    in_coords, out_counts = make_coords()

    # define the number of threads
    n_threads = 4

    # define the number of blocks of 8 floats vector per rows
    n_elem = len(in_coords[0,:]) / 8

    with Timer() as t:
        mandelbrot.mandelbrot_multrithreads_ILP(in_coords, out_counts, n_threads, n_elem,1024 )
    seconds = t.interval

    print("{} Million Complex FMAs in {} seconds, {} million Complex FMAs / second".format(out_counts.sum() / 1e6, seconds, (out_counts.sum() / seconds) / 1e6))

    plt.imshow(np.log(out_counts))
    plt.savefig('plot_p3.png')
    plt.show()

