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

    """in_coords = np.asarray([0.2 + 0.5j, 0.2 + 0.5j, 0.2 + 0.5j, 0.2 + 0.5j, 0.2 + 0.5j, 0.2 + 0.5j, 0.2 + 0.5j, 0.2 + 0.5j]).astype(np.complex64).reshape(1,8)
    out_counts = np.zeros_like(in_coords, dtype=np.uint32)
    print in_coords.shape"""
    for threads in [1, 2, 4]:
        with Timer() as t:
            mandelbrot.mandelbrot(in_coords, out_counts, threads, 1024)
        seconds = t.interval

        print("{} threads: {} Million Complex FMAs in {} seconds, {} million Complex FMAs / second".format(threads, out_counts.sum() / 1e6, seconds, (out_counts.sum() / seconds) / 1e6))
        #print out_counts[19:20, 12:20], out_counts[2000:2001, 2000:2008], out_counts[3500:3501, 3500:3508]
        plt.imshow(np.log(out_counts))
        plt.show()
