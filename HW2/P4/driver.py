###########################################################################
#Jaemin Cheun
#CS205, Fall 2015 Computing Foundations for Computer Science
#Nov 4, 2015
#driver.py
###########################################################################

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

    # initialize threads
    threads = [None] * num_threads

    # initialize events: each event corresponds to a thread,iteration pair. Also change it into a numpy array 
    events = np.array([[threading.Event() for i in range(iterations)] for j in range(num_threads)])  

    # Start threads
    for idx in range(num_threads):
        threads[idx] = threading.Thread(target = filtering_async, args = (idx, tmpA, tmpB, events, num_threads, iterations))
        threads[idx].start()

    #Terminate all threads
    for thread in threads:
        thread.join()

    return tmpA

# This function allow threads to begin an iteration of filtering asynchronously
def filtering_async(thread_idx, tmpA, tmpB, events, num_threads, iterations):
    for i in range(iterations):
        #Check cases such as thread index is 0 or thread index is the last one
        if num_threads > 1 and i > 0:
            
            # thread_idx - 1 doesn't exist
            if thread_idx == 0: 
                events[thread_idx + 1, i - 1].wait()

            # thread_idx + 1 doesn't exist
            elif thread_idx == num_threads - 1:
                events[thread_idx - 1, i - 1].wait()

            else:
                events[thread_idx - 1, i - 1].wait()
                events[thread_idx + 1, i - 1].wait()

        filtering.median_3x3(tmpA, tmpB, thread_idx, num_threads)
        events[thread_idx, i].set()

        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA


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
