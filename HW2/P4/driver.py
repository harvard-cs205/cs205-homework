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
import time

#Need to make sure that iteration i has completed (for at least threads n-1, n, n+1) before moving onto iteration i+1 for thread n
#Solution: Once one thread has completed iteration i, block until all threads have completed iteration i
def iterate_3x3(iterations, tmpA, tmpB, thread, num_threads, events):
    for i in range(iterations):
        filtering.median_3x3(tmpA, tmpB, thread, num_threads)
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA
        print "setting " + str(thread) + " in iteration " + str(i)
        events[i][thread].set()
        for j in range(num_threads):
            #print "waiting for event " + str(j) + " on thread " + str(thread) + " in iteration " + str(i)
            events[i][j].wait()
            #print "done waiting for event " + str(j) + " on thread " + str(thread) + " in iteration " + str(i)


def py_median_3x3(image, iterations, num_threads):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)
    threads = []
    events = []

    for iteration in range(iterations):
        events_row = []
        for thread in range(num_threads):
            e = threading.Event()
            events_row.append(e)
        events.append(events_row)
    
    for thread in range(num_threads):
        t = threading.Thread(target=iterate_3x3, args=(iterations, tmpA, tmpB, thread, num_threads, events))
        threads.append(t)
        t.start()
    for thread in threads:
        thread.join()
    del events
    del threads
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

    for thread_count in [1, 2, 4, 8]:
        with Timer() as t:
            new_image = py_median_3x3(input_image, 10, thread_count)

        pylab.figure()
        pylab.imshow(new_image[1200:1800, 3000:3500])
        pylab.title('after - zoom {} threads'.format(thread_count))

        print("{} threads: {} seconds for 10 filter passes.".format(thread_count, t.interval))
    pylab.show()
