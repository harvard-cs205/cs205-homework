import sys
import os.path
sys.path.append(os.path.join('..', 'util'))

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import numpy as np
import pylab

import filtering
from timer import Timer
import threading
from multiprocessing.pool import ThreadPool

def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmp_even = image.copy()
    tmp_odd = np.empty_like(tmp_even)

    # create pool of correct number of threads
    pool = ThreadPool(num_threads)

    # set up event for each thread for each iteration
    events = [[threading.Event() for k in range(iterations)] for j in range(num_threads)]

    def median_wrapper(thread_n, num_threads=num_threads, iterations=iterations, tmpA=tmp_even, tmpB=tmp_odd, events=events):
        ''' Helper function for each thread '''
        for i in range(iterations):
            # wait for completion of previous iteration
            if i > 0:
                events[thread_n][i - 1].wait()
                if thread_n > 0:
                    events[thread_n - 1][i - 1].wait()
                if thread_n < num_threads - 1:
                    events[thread_n + 1][i - 1].wait()

            # median filter on every Nth line, starting with line n
            filtering.median_3x3(tmpA, tmpB, thread_n, num_threads)

            # set event saying we finished this iteration
            events[thread_n][i].set()

            # swap direction of filtering
            tmpA, tmpB = tmpB, tmpA

    # send work to all the threads
    pool.map(median_wrapper, range(num_threads), chunksize=1)

    return tmp_even if iterations % 2 == 0 else tmp_odd

def numpy_median(image, iterations=10):
    ''' filter using numpy '''
    for i in range(iterations):
        padded = np.pad(image, 1, mode='edge')
        stacked = np.dstack((padded[:-2,  :-2], padded[:-2,  1:-1], padded[:-2,  2:],
                             padded[1:-1, :-2], padded[1:-1, 1:-1], padded[1:-1, 2:],
                             padded[2:,   :-2], padded[2:,   1:-1], padded[2:,   2:]))
        image = np.median(stacked, axis=2)

    return image


if __name__ == '__main__':
    input_image = np.load('image.npz')['image'].astype(np.float32)

   #pylab.gray()

   #pylab.imshow(input_image)
   #pylab.title('original image')

   #pylab.figure()
   #pylab.imshow(input_image[1200:1800, 3000:3500])
   #pylab.title('before - zoom')

    # verify correctness
    from_cython = py_median_3x3(input_image, 2, 5)
    from_numpy = numpy_median(input_image, 2)
    assert np.all(from_cython == from_numpy)

    for nt in [1, 2, 4]:
        with Timer() as t:
            new_image = py_median_3x3(input_image, 10, nt)

       #pylab.figure()
       #pylab.imshow(new_image[1200:1800, 3000:3500])
       #pylab.title('after - zoom')

        print("{0} seconds for 10 filter passes with {1} threads.".format(t.interval, nt))
       #pylab.show()
