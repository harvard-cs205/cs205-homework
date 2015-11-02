# driver.py

######################
#
# Submission by Kendrick Lo (Harvard ID: 70984997) for
# CS 205 - Computing Foundations for Computational Science (Prof. R. Jones)
# 
# Homework 2 - Problem 4
#
######################

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

def median_loop(A, B, iters, thr, step, e):
    ''' Helper function to call multiple iterations of median filter. 

    Synchronous version.

    A, B - initial and modified image data arrays
    iters - number of filter passes
    thr - thread index
    step - number of rows to skip = number of threads
    e - Threading Event array
    '''

    for i in range(iters):
        e[thr, i].clear()  # reset flag
        filtering.median_3x3(A, B, thr, step)
        A, B = B, A
        e[thr, i].set()  # set thread's flag for iteration i -> "i am done"

        for k in range(step):
            if k!=thr:
                e[k, i].wait()  # wait for all other threads to finish

    return A

def async_median_loop(A, B, iters, thr, step, e):
    ''' Helper function to call multiple iterations of median filter. 

    Asynchronous version.

    A, B - initial and modified image data arrays
    iters - number of filter passes
    thr - thread index
    step - number of rows to skip = number of threads
    e - Threading Event array
    '''

    for i in range(iters):
        e[thr, i].clear()  # reset flag
        filtering.median_3x3(A, B, thr, step)
        A, B = B, A
        e[thr, i].set()  # set thread's flag for iteration i -> "i am done"

        if step==1:
            # do not wait, only one thread
            pass 
        elif step==2:
            # only two threads, wait for other thread
            if thr==0:  
                e[1, i].wait()
            else:
                e[0, i].wait()
        else:
            # find out who neighbors are                
            l_neighbor = thr - 1  
            r_neighbor = thr + 1
            if l_neighbor<0:
                l_neighbor = step - 1  # wraparound
            if r_neighbor>(step-1):
                r_neighbor = 0  # wraparound

            e[l_neighbor, i].wait()  # } wait (only) for two neighboring 
            e[r_neighbor, i].wait()  # } threads to finish

    return A


def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    threadlist = []

    # create an array of Event objects
    # to flag when each thread is done with a particular iteration
    e = np.empty([num_threads, iterations], dtype=object)
    for i in range(num_threads):
        for j in range(iterations):
            e[i, j] = threading.Event()

    # construct loop to start up each thread to begin work
    #
    # target can be set to `median_loop`, where threads cannot start
    # an iteration of filtering before the data for that iteration is ready;
    # or to `async_median_loop`, where thread can continue once neighboring
    # threads have finished
    for threadidx in range(num_threads):
        th = threading.Thread(target=async_median_loop,
                              args=(tmpA, tmpB, 
                                    iterations, threadidx, num_threads, e))
        threadlist.append(th)
        th.start()

    for thread in threadlist:
        thread.join()  # wait for all threads to end before program completes

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

    num_iters = [10]  # can expand list to test for different iterations
    threads = [1, 2, 4, 8]  # can modify list to test for different thread numbers

    for its in num_iters:

        for thr in threads:

            # verify correctness
            from_cython = py_median_3x3(input_image, 2, 5)
            from_numpy = numpy_median(input_image, 2)
            assert np.all(from_cython == from_numpy)

            with Timer() as t:
                new_image = py_median_3x3(input_image, its, thr)

            pylab.figure()
            pylab.imshow(new_image[1200:1800, 3000:3500])
            pylab.title('after - zoom')

            print("Number of threads: {}".format(thr))
            print("{} seconds for {} filter passes.".format(t.interval, its))
            pylab.show()
