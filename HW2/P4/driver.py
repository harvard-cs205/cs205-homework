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

def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    #we will create an event for each iteration in each thread
    #for each iteration, we will launch a thread
    #   we will pass that thread the two images, as well as the list of events
    events = [[threading.Event() for i in range(iterations)] for j in range(num_threads)]    

    #to compute iteration i for thread n, threads n, n-1, and n+1 must have completed 
    #   iteration i-1    
    def run_thread(thread):
        for i in range(iterations):
            #wait for thread n on iteration i-1
            events[thread][i-1].wait()

            #wait for thread n-1 on i-1
            if thread > 0:
                events[thread-1][i-1].wait()

            #wait for thread n_1 on i-1
            if thread < (num_threads-1):
                events[thread+1][i-1].wait()

            #filter
            filtering.median_3x3(tmpA, tmpB, thread, num_threads)

            #swap
            tmpA, tmpB = tmpB, tmpA

            #unpause our current thread
            events[thread][i].set()

    
    threads = [threading.Thread(target=run_thread, args=(i)) for i in range(num_threads)]
    
    for i in threads:
        i.start()

    for i in threads:
        i.join()


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
