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

logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s] (%(threadName)-10s) %(message)s',
                    )


def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    threads = []

    # Create events that signals the threads to avoid collisions
    events = np.array([threading.Event()]*(iterations*num_threads)).reshape([num_threads, iterations])

    if num_threads==1:
        for i in range(iterations):
            filtering.median_3x3(tmpA, tmpB, 0, 1)
            # swap direction of filtering
            tmpA, tmpB = tmpB, tmpA
    else:
        # initialize the threads
        for tid in range(num_threads):
            t = threading.Thread(target=filterHelper, 
                                 args=(iterations, tid, num_threads, tmpA, tmpB, events))
            threads.append(t)
            t.start()
        map(lambda t: t.join(), threads)

    return tmpA

def filterHelper(iterations, tid, num_threads, tmpA, tmpB, events):
    ''' Function that coordinates the filtering on each line of pixels'''
    # The first iteration only sets events, no more waiting
    filtering.median_3x3(tmpA, tmpB, tid, num_threads)
    # swap direction of filtering
    tmpA, tmpB = tmpB, tmpA
    events[tid, 1].set()

    for i in range(1, iterations):
        #Handle events
        if tid>0:
            events[tid-1, i-1].wait()
        if tid<num_threads-1:
            events[tid+1, i-1].wait()

        filtering.median_3x3(tmpA, tmpB, tid, num_threads)
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA
        events[tid, i].set()



def py_median_3x3_original(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    for i in range(iterations):
        filtering.median_3x3(tmpA, tmpB, 0, 1)
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA

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
        new_image = py_median_3x3(input_image, 10, 5)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
