# set up easy cython import
import numpy as np
import pyximport, os
from distutils.extension import Extension 
ext_modules = [Extension('_mandelbrot', ['_mandelbrot.pyx'],
                          extra_compile_args=['-fopenmp', '-O3', '-march=native'], 
                          extra_link_args=['-fopenmp'])] 
pyximport.install(setup_args={"include_dirs": [np.get_include(), os.curdir], 'ext_modules': ext_modules})
from _mandelbrot import mandelbrot

# a helpful timer class that can be used by the "with" statement
import time
class Timer(object):
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

# create coordinates, along with output count array
def make_coords(center=(-0.575 - 0.575j),
                width=0.0025,
                count=4000):

    x = np.linspace(start=(-width / 2), stop=(width / 2), num=count)
    xx = center + (x + 1j * x[:, np.newaxis]).astype(np.complex64)
    return xx, np.zeros_like(xx, dtype=np.uint32)
