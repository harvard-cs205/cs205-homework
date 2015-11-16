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
    thread_pos = [1,2,4]
    repetitions = 6
    x_threads = []
    FMAs_per_second = []
    for number_of_threads in thread_pos:
        for _ in range(repetitions):
            in_coords, out_counts = make_coords()
            num_threads = 2
            with Timer() as t:
                mandelbrot.mandelbrot(in_coords, out_counts, number_of_threads, 1024)
            seconds = t.interval
            x_threads.append(number_of_threads)
            FMAs_per_second.append((out_counts.sum() / seconds) / 1e6)
            print("{} Threads: {} Million Complex FMAs in {} seconds, {} million Complex FMAs / second".format(number_of_threads, out_counts.sum() / 1e6, seconds, (out_counts.sum() / seconds) / 1e6))
            #plt.imshow(np.log(out_counts))
            #plt.show()
    plt.figure(1)
    plt.scatter(x_threads, FMAs_per_second)
    plt.xlabel('Number of threads')
    plt.ylabel('Million Complex FMAs per second')
    plt.title('Performance With Instruction-Level Parallelism')
    plt.show()

    
