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


def do_filtering(thread_id, tmpA, tmpB, num_threads, iterations, events):
    """
    Since median filter relies on the pixels around it, each thread cannot do
    a second pass before the first pass is done around it.

    This function is passed to each thread as the "target," and is run
    as part of the thread's `run` method. It does a pass of filtering and
    then waits at each step for the previous and next threads to be done with
    their computation before continuing to the next one.
    """
    neighbors = []
    if thread_id > 0:
        neighbors.append(thread_id - 1)
    if thread_id < num_threads - 1:
        neighbors.append(thread_id + 1)

    for iteration in xrange(iterations):
        prev_iter = iteration - 1

        # if this is the first iteration, we don't need to worry
        if iteration != 0:

            # check to make sure neighbors are done with the previous pass
            for n in neighbors:
                events[n][prev_iter].wait()

        # do some filtering
        filtering.median_3x3(tmpA, tmpB, thread_id, num_threads)

        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA

        # signal that we're done with this iteration
        events[thread_id][iteration].set()


def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    # create a list of events, one for each iteration, for each thread
    events = [[threading.Event() for _ in xrange(iterations)] for __ in xrange(num_threads)]

    # create the thread objects
    threads = [threading.Thread(target=do_filtering, args=[i, tmpA, tmpB, num_threads, iterations, events]) for i in xrange(num_threads)]

    # start the computation
    for x in threads:
        x.start()

    # block the master thread until all worker threads terminate
    for x in threads:
        x.join()

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

    times = []
    times_orig = []

    threads = 4

    with Timer() as t:
        new_image = py_median_3x3(input_image, 10, threads)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
