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
    # test_coords = in_coords.astype(np.float32)

    # test_coords_in = in_coords
    # test_coords_out = out_counts
    #test = mandelbrot.example_sqrt_8(test_coords_in, test_coords_out)

    with Timer() as t:
        mandelbrot.mandelbrot(in_coords, out_counts)
    seconds = t.interval

    print("{} Million Complex FMAs in {} seconds, {} million Complex FMAs / second".format(out_counts.sum() / 1e6, seconds, (out_counts.sum() / seconds) / 1e6))

    plt.imshow(np.log(out_counts))
    plt.show()
    
    # print 'test', test

    
    # with Timer() as t:
    #     mandelbrot.mandelbrot(in_coords, out_counts, 1024)
    # seconds = t.interval


    # plt.imshow(np.log(out_counts))
    # plt.show()
    

