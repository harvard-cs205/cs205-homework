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

def target_function(tmpA, tmpB, iterations, events, num_threads, thread_no):
    for i in range(iterations):
        filtering.median_3x3(tmpA, tmpB, thread_no, num_threads)
        # if our thread index is greater than 0, wait for thread no -1 iteration i-1
        if thread_no > 0 and i > 0:
            thread_min_one = (thread_no - 1) % num_threads
            thread_plus_one = (thread_no + 1) % num_threads
            events[i-1][thread_min_one].wait()
            events[i-1][thread_plus_one].wait()

        # set this event as finished
        events[i][thread_no].set()
        # Swap like it was originally
        tmpA, tmpB = tmpB, tmpA




def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)
    threads = []
    events = []
    # Create a bunch of events
    for i in range(iterations):
        events.append([threading.Event() for j in range(num_threads)])

    for i in range(num_threads):
        new_thread = threading.Thread(target=target_function, args=(tmpA, tmpB, iterations, events, num_threads, i))
        threads.append(new_thread)
        new_thread.start()

    # Make sure we wait for all the threads to finish
    for thread in threads:
        thread.join()

    # for i in range(iterations):
    #     filtering.median_3x3(tmpA, tmpB, 0, 1)
    #     # swap direction of filtering
    #     tmpA, tmpB = tmpB, tmpA

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

    for num_threads in [1,2,4]:
        with Timer() as t:
            new_image = py_median_3x3(input_image, 10, num_threads)

        pylab.figure()
        pylab.imshow(new_image[1200:1800, 3000:3500])
        pylab.title('after - zoom')

        print("{} seconds for 10 filter passes. number of threads{}".format(t.interval, num_threads))
        pylab.show()
