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

def worker(tmpA, tmpB, iterations, threadidx, num_threads, events):

    for i in range(iterations):
        # we care about events only when there are more than one thread
        if num_threads>1:
            # no wait at 1st iteration
            if i > 0 :
                # if first line just wait for the next one
                if threadidx == 0:
                    events[threadidx+1][i-1].wait()
                # if last line just wait for the one before
                elif threadidx == num_threads-1:
                    events[threadidx-1][i-1].wait()
                # else wait for line before and after
                else :
                    events[threadidx+1][i-1].wait()
                    events[threadidx-1][i-1].wait()

        #have each thread work on every num_threads-th thread
        filtering.median_3x3(tmpA, tmpB, threadidx, num_threads)
        # swap direction of filtering (change the pointers)
        tmpA, tmpB = tmpB, tmpA
        #awakes all the thread waiting for it
        if num_threads>1:
            events[threadidx][i].set()

def py_median_3x3(image, iterations=10, num_threads=1):
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    #Initialize the events one event per (threadid, iteration step) tuple
    events = [[threading.Event() for _ in range(iterations)] for _ in range(num_threads)]

    #Initialize create a list of threads
    thread_list=[]
    for threadidx in range(num_threads):
        #create num_threads one by one, note that we pass threadidx which is the index of the thread
        th = threading.Thread(target = worker, args = (tmpA, tmpB, iterations, threadidx, num_threads, events))
        thread_list.append(th)
        th.start()

    # make sure it gets the results
    for th in thread_list:
        th.join()

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
        new_image = py_median_3x3(input_image, 10, 8)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
