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


def py_median_3x3_single(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    for i in range(iterations):
        filtering.median_3x3(tmpA, tmpB, 0, 1)
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA

    return tmpA


def py_median_3x3_sync(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    for i in range(iterations):
        threads = []
        for j in xrange(num_threads):
            t = threading.Thread(target=filtering.median_3x3, args=(tmpA, tmpB,
                                 j, num_threads))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA

    return tmpA


def py_median_3x3_async(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    # List of event for each iteration & thread
    events = []
    for i in xrange(iterations):
        events_list = [threading.Event()]
        # The first row is a virtual one always set (because the first thread will wait for it)
        events_list[0].set()

        for j in xrange(num_threads + 1):
            events_list.append(threading.Event())
        # The last row is a virtual one always set (because the last thread will wait for it)
        events_list[-1].set()
        events.append(events_list)

    def async_median(tmpA, tmpB, offset, step, iterations, events):
        # First iteration
        filtering.median_3x3(tmpA, tmpB, offset, step)

        tmpA, tmpB = tmpB, tmpA
        events[0][offset + 1].set()

        for i in xrange(1, iterations):
            # The current has the position offset + 1 in the list

            events[i - 1][offset].wait()
            events[i - 1][offset + 2].wait()

            # Filter computation
            filtering.median_3x3(tmpA, tmpB, offset, step)
            tmpA, tmpB = tmpB, tmpA

            events[i][offset + 1].set()

    threads = []

    for j in xrange(num_threads):
        t = threading.Thread(target=async_median, args=(tmpA, tmpB,
                             j, num_threads, iterations, events))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
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

    with Timer() as t:
        from_single = py_median_3x3_single(input_image, 2, 5)
    print('Serial version in {} seconds'.format(t.interval))

    with Timer() as t:
        from_numpy = numpy_median(input_image, 2)
    print('Numpy version in {} seconds'.format(t.interval))

    for num in [1, 2, 4]:
        with Timer() as t:
            from_sync = py_median_3x3_sync(input_image, 10, num)
        print('Sync version in {} seconds with {} threads.'.format(t.interval,
              num))

        with Timer() as t:
            from_async = py_median_3x3_async(input_image, 10, num)
        print('Async version in {} seconds with {} threads.'.format(t.interval,
              num))

    # verify correctness
    assert np.all(from_single == from_numpy)
    assert np.all(from_sync == from_numpy)
    assert np.all(from_async == from_numpy)

    with Timer() as t:
        new_image = py_median_3x3_async(input_image, 10, 8)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
