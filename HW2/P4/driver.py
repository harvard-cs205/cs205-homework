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

def worker(tmpA, tmpB, threadidx, num_threads):

        #have each thread work on every num_threads-th thread
        filtering.median_3x3(tmpA, tmpB, threadidx, num_threads)

def py_median_3x3(image, iterations=10, num_threads=1):
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    #Initialize the events one event per (threadid, iteration step) tuple
    events = [[threading.Event() for _ in range(iterations)] for _ in range(num_threads)]

    for _ in range(iterations):
        #Initialize create a list of threads
        thread_list=[]
        for threadidx in range(num_threads):
            #create num_threads one by one, note that we pass threadidx which is the index of the thread
            th = threading.Thread(target = worker, args = (tmpA, tmpB, threadidx, num_threads))
            thread_list.append(th)
            th.start()

        # make sure it gets the results
        for th in thread_list:
            th.join()

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

    print("Synchronized")
    for N in [1,2,4]:

        with Timer() as t:
            new_image = py_median_3x3(input_image, 10, N)

        # pylab.figure()
        # pylab.imshow(new_image[1200:1800, 3000:3500])
        # pylab.title('after - zoom')
        print("With {} threads".format(N))
        print("{} seconds for 10 filter passes.  ".format(t.interval))
        # pylab.show()
