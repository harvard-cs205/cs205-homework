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

    for i in range(iterations):
        filtering.median_3x3(tmpA, tmpB, 0, 1)
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA

    return tmpA


def worker(events, thread_id, thread_num, iterations, orig_imag, new_imag):
    for i in range(iterations):
        
        # Do nothing for the first iteration, however still needed for i-1
        if i == 0:
            pass

        # Do nothing if there is only one thread
        elif thread_num == 1:
            pass
        
        # Corner case first thread only has 1 dependent thread
        elif thread_id == 0:
            events[thread_id+1, i-1].wait()
        
        # Corner case last thread only has 1 dependent thread
        elif thread_id == thread_num-1:
            events[thread_id-1, i-1].wait()

        # All other threads have 2 dependents 
        # Tell both previous and next to wait on current thread
        else:
            events[thread_id-1, i-1].wait()
            events[thread_id+1, i-1].wait()

        # Run provided cython function to filter
        filtering.median_3x3(orig_imag, new_imag, thread_id, thread_num)

        # Set current thread ID so that dependent threads know to wait
        events[thread_id, i].set()

        # Simultaneous change original image and new filtered image
        orig_imag, new_imag = new_imag, orig_imag

def py_median_3x3_threaded(image, iterations, num_threads):
    ''' filter with a 3x3 median threading '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    # Create the event array for all threads and every iteration
    events = np.array([[threading.Event()]*iterations]*num_threads)

    # Create Threads given the worker function and start
    for t in range(num_threads):
        threads = threading.Thread(target=worker, args=(events, t, num_threads, iterations, tmpA, tmpB))
        threads.start()
    
    # Join Threads to ensure all threads have completed their work
    threads.join()

    # If odd number of interations return empty image
    if iterations % 2 == 1:
        return tmpB
    else:
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

    # pylab.gray()

    # pylab.imshow(input_image)
    # pylab.title('original image')

    # pylab.figure()
    # pylab.imshow(input_image[1200:1800, 3000:3500])
    # pylab.title('before - zoom')

    # verify correctness
    # from_cython = py_median_3x3(input_image, 2, 5)
    # from_numpy = numpy_median(input_image, 2)
    # assert np.all(from_cython == from_numpy)

    with Timer() as t:
        new_image = py_median_3x3(input_image, 10, 1)

    # pylab.figure()
    # pylab.imshow(new_image[1200:1800, 3000:3500])
    # pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    # pylab.show()

    ### My Code ###
    from_cython = py_median_3x3_threaded(input_image, 2, 5)
    from_numpy = numpy_median(input_image, 2)
    assert np.all(from_cython == from_numpy), "Images don't match!"

    with Timer() as time1:
        threaded_image = py_median_3x3_threaded(input_image, 10, 1)

    with Timer() as time2:
        threaded_image = py_median_3x3_threaded(input_image, 10, 2)
    
    with Timer() as time4:
        threaded_image = py_median_3x3_threaded(input_image, 10, 4)

    results = [(1, time1), (2, time2), (4, time4)]

    for t, r in results:
        print("{} seconds for 10 filter passes and {} threads".format(r.interval, t))


