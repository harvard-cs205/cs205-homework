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

def parallel_filter(imgA, imgB, tid, num_threads, events, iterations):
    for i in range(iterations):
        # for each thread on iteration i, make sure that:
        # iteration i - 1 for threads n, n - 1, and n + 1 are done
        if num_threads > 1 and i > 0:
            events[(tid - 1) % num_threads][i - 1].wait()
            events[(tid + 1) % num_threads][i - 1].wait()

        # do work, after ready
        filtering.median_3x3(imgA, imgB, tid, num_threads)
        
        # mark that iteration as done
        events[tid][i].set()

        # swap direction
        imgA, imgB = imgB, imgA

def py_median_3x3(image, num_threads, iterations=10):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    # keep track of finished iterations for each thread
    events = [[threading.Event() for i in range(iterations)] for j in range(num_threads)]

    # keep threads in list for easy joining later
    # start each thread's job
    threads = []
    for tid in range(num_threads):
        threads.append(threading.Thread(target=parallel_filter,
                                        args=(tmpA, tmpB, tid, num_threads,
                                              events, iterations)))
        threads[tid].start()

    # finish threads
    for thread in threads:
        thread.join()

    return tmpA

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

    pylab.gray()

    pylab.imshow(input_image)
    pylab.title('original image')

    pylab.figure()
    pylab.imshow(input_image[1200:1800, 3000:3500])
    pylab.title('before - zoom')

    # verify correctness
    from_cython = py_median_3x3(input_image, 1, 2)
    from_numpy = numpy_median(input_image, 2)
    assert np.all(from_cython == from_numpy)

    with Timer() as t:
        new_image = py_median_3x3(input_image, 4, 10)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
