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


def median_worker(tmpA, tmpB, iters, cur_thread, num_threads, events): 
    # in each iteration, we need to wait for the previous iterations 
    for i in range(iters):
        # call wait on the adjacent threads previous iteration
        if i > 0:
            # wait until we have finished previous iteration
            events[cur_thread][i - 1].wait()
            if cur_thread > 0: 
                # wait until neighbors finish previous iteration
                events[cur_thread - 1][i - 1].wait()
            if cur_thread < num_threads - 1:
                events[cur_thread + 1][i - 1].wait()
        # perform filtering 
        filtering.median_3x3(tmpA, tmpB, cur_thread, num_threads)
        # swap tmpA and tmp B
        tmpB, tmpA = tmpA, tmpB 
        # we have finished the current computation
        events[cur_thread][i].set()

def py_median_3x3(image, iters, num_threads):
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)
    # create an event for eacch thread/iteration
    # will be set when computation for thread/iteration is finished 
    events  = [[threading.Event() for i in range(iters)] for j in range(num_threads)]
    threads = [threading.Thread(target=median_worker, args=(tmpA, tmpB, iters, ti, num_threads, events)) for ti in range(num_threads)]
    # start the therads
    for th in threads:
        th.start()
    # wait for all computations to tinish before returning
    for th in threads:
        th.join()
    return tmpA 

def old_py_median_3x3(image, iterations=10, num_threads=1):
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
    
    # we need to compare w/ different number of threads s
    n_threads = 4
    with Timer() as t:
        new_image = py_median_3x3(input_image, 10, n_threads)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
