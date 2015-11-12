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

# Helper function to make sure we don't wait
# forever on the first iteration or break on the last one.
def bounds_check(j, num_threads):
    return j >= 0 and j < num_threads

def median_worker(tmpa, tmpb, j, events, num_threads, iters):
    for i in range(iters):
        # Wait on all the necessary conditions in order to execute (extra credit).
        if bounds_check(j - 1, num_threads) and i - 1 >= 0:
            events[i - 1][j - 1].wait()
        if bounds_check(j, num_threads) and i - 1 >= 0:
            events[i - 1][j].wait()
        if bounds_check(j + 1, num_threads) and i - 1 >= 0:
            events[i - 1][j + 1].wait()
        filtering.median_3x3(tmpa, tmpb, j, num_threads)

        # You are now done, so set that event flag.
        events[i][j].set()
        tmpa, tmpb = tmpb, tmpa

def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    # Declare the set of all events: one for each thread for each iteration.
    events = [[threading.Event() for i in range(num_threads)] for k in range(iterations)]

    threads = [threading.Thread(target=median_worker, args = \
        (tmpA, tmpB, j, events, num_threads, iterations)) for j in range(num_threads)]

    # Launch all of the worker threads.
    for thread in threads:
        thread.start()

    # Wait for all of the worker threads.
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
    from_cython = py_median_3x3(input_image, 2, 5)
    from_numpy = numpy_median(input_image, 2)
    assert np.all(from_cython == from_numpy)

    with Timer() as t:
        new_image = py_median_3x3(input_image, 10, 4)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
