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

def thread_helper(events, iterations, thread, tmpA, tmpB, num_threads):
    for iter in range(iterations):
        # completed thread n, n-1, n+1 for iter i to start thread n iter i
        if iter > 0: 
            # wait for all threads nearby to finish iteration i-1
            events[thread][iter-1].wait() 
            if thread > 0:
                events[thread-1][iter-1].wait()
            if thread < num_threads-1:
                events[thread+1][iter-1].wait() 

        # start from thread, filter every N lines
        filtering.median_3x3(tmpA, tmpB, thread, num_threads)

        tmpA, tmpB = tmpB, tmpA

        # finish this iteration
        events[thread][iter].set()

        # swap direction of filtering
        #mpA, tmpB = tmpB, tmpA


def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    # create event list for each thread in each iteration
    events = [[threading.Event() for i in range(iterations)] for j in range(num_threads)] 
    # create a list of threads
    threads = []

    # send workers to all threads
    for thread in range(num_threads):
        threads.append(threading.Thread(target=thread_helper, args=(events, iterations, thread, tmpA, tmpB, num_threads)))
        threads[thread].start()

    # wait all threads to finish work
    for thread in range(num_threads):
        threads[thread].join()

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
    for thread_n  in [1,2,4]: 

        from_cython = py_median_3x3(input_image, 10, thread_n)
        from_numpy = numpy_median(input_image, 10)
        assert np.all(from_cython == from_numpy)

        with Timer() as t:
            new_image = py_median_3x3(input_image, 10, thread_n)
        print("{} seconds for 10 filter passes.".format(t.interval))

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    #print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
