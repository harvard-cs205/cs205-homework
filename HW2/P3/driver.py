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

    #############################
    # Multithreading experiment #
    #############################

    threadcount = [1, 2, 4]

    execution_times = []

    for n in threadcount:
        with Timer() as t:
            mandelbrot.mandelbrot(in_coords, out_counts, 1024, num_threads=n)
        seconds = t.interval
        execution_times += [seconds]

        print("{} Million Complex FMAs in {} seconds, {} million Complex FMAs / second".format(out_counts.sum() / 1e6, seconds, (out_counts.sum() / seconds) / 1e6))

        plt.imshow(np.log(out_counts))
        plt.show()

    f, ax = plt.subplots(1)
    plt.plot(threadcount, execution_times)
    plt.xlabel('N')
    plt.ylabel('Time (sec.)')
    plt.title('Execution time for 1, 2 and 4 threads with AVX')
    ax.set_ylim(0,)
    plt.show()
