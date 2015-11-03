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
import logging

# https://pymotw.com/2/threading/
logging.basicConfig(level=logging.INFO,
                    format='(%(threadName)-10s) %(message)s',)

def worker_from_second_itr(i, n, num_threads, iterations, tmpA, tmpB, events):
    for itr in range(1, iterations):
        # wait for neighboring events to finish
        prec_i = i - 1 if i > 0 else (num_threads - 1)
        succ_i = i + 1 if i < (num_threads - 1) else 0
        # @x: wait for neighbors
        logging.debug('thread %d->itr %d: waiting for neighbors', i, itr)
        events[itr - 1][prec_i].wait()
        events[itr - 1][i].wait()
        events[itr - 1][succ_i].wait()
        # @x: filter
        logging.debug('thread %d->itr %d: get event signal; filters', i, itr)
        filtering.median_3x3(tmpA, tmpB, i, num_threads)
        # @x: swap i th
        logging.debug('thread %d->itr %d: finish filtering; swap', i, itr)
        for k in range(0, n, num_threads):
            tmpA[k], tmpB[k] = tmpB[k], tmpA[k]
        # @x: signal waiting threads
        logging.debug('thread %d->itr %d: finish swap; set event', i, itr)
        events[itr][i].set()


def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)
    if iterations == 0:
        return tmpA
    n = tmpA.shape[0]
    # iterations x num_threads matrix
    logging.debug('num_threads = %d', num_threads)
    logging.debug('iterations = %d', iterations)
    events = [[threading.Event()] * num_threads] * iterations
    # iterate once first
    filtering.median_3x3(tmpA, tmpB, 0, 1)
    tmpA, tmpB = tmpB, tmpA
    for k in range(len(events[0])):
        e = events[0][k]
        e.set()
        logging.debug('thread %d->itr %d: finish swap; set event', k, 0)
    for i in range(num_threads):
        t = threading.Thread(target=worker_from_second_itr, args=(i, n, num_threads, iterations, tmpA, tmpB, events))
        logging.debug('start thread  %d', i)
        t.start()
    # for i in range(iterations):
    #     filtering.median_3x3(tmpA, tmpB, 0, 1)
    #     # swap direction of filtering
    #     tmpA, tmpB = tmpB, tmpA

    main_thread = threading.currentThread()
    for t in threading.enumerate():
        if t is main_thread:
            continue
        logging.debug('joining %s', t.getName())
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

    # verify correctness
    from_cython = py_median_3x3(input_image, 2, 5)
    from_numpy = numpy_median(input_image, 2)
    # assert np.all(from_cython == from_numpy)

    # @x: test different number of threads
    for num_threads in [1, 2, 4]:
        with Timer() as t:
            new_image = py_median_3x3(input_image, 10, num_threads)
        print("{0} seconds for 10 filter passes with {1} threads.".format(t.interval, num_threads))

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    # print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
