import sys
import os.path
sys.path.append(os.path.join('..', 'util'))

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import numpy as np
import matplotlib.pyplot as pylab

import filtering
from timer import Timer
import threading

def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)


    # we will have one event for each thread, each iteration
    event_list = [[threading.Event() for _ in range(iterations)] for _ in range(num_threads)]

    all_threads = [] 

    # create num_thread threads, and each will process a different chunk of the image
    for thread_id in range(num_threads): 
        thread = threading.Thread(target=do_work, args=(thread_id, num_threads, tmpA, tmpB, event_list, iterations))
        all_threads.append(thread) 
        thread.start()

    for t in all_threads:
        t.join()

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

def do_work(thread_id, num_threads, tmpA, tmpB, event_list, iterations=10):
    # unless this is the first iteration, we need to wait for a couple events to finish first 
    for iteration in range(iterations):
        print "Thread id: %d, iteration: %d " % (thread_id, iteration)
        # 0th iteration does not require any waiting 
        if iteration == 0: 
            pass
        elif thread_id > 0: 
            print "Waiting on thread_id=%d, iterations=%d" % (thread_id, iteration)
            event_list[thread_id-1][iteration-1].wait() 
        elif thread_id < num_threads - 1: 
            print "Waiting on thread_id=%d, iterations=%d" % (thread_id, iteration)
            event_list[thread_id+1][iteration-1].wait()

        filtering.median_3x3(tmpA, tmpB, thread_id, num_threads)
        # broadcast that our current iteration's filtering is now done 
        event_list[thread_id][iteration].set()
        print "thread_id, iteration = (%d, %d) has been set" % (thread_id, iteration)
        tmpA, tmpB = tmpB, tmpA


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
    print from_cython[0:10, 0:10] - from_numpy[0:10, 0:10]
    assert np.all(from_cython == from_numpy)

    with Timer() as t:
        new_image = py_median_3x3(input_image, 10, 8)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
