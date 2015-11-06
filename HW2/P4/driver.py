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

    thread_list = [] 
    event_array = []

    # make event array : [iteration(row) * thread_idx(column)]
    for i in range(iterations):
        row = []
        for j in range(num_threads):
            row.append(threading.Event())
        event_array.append(row)


    # make N threads
    for thread_idx in range(num_threads):
        thread = threading.Thread(target=image_processing_multithread,
                                  args=(thread_idx, iterations, num_threads, event_array, tmpA, tmpB))
        thread_list.append(thread) # to terminate threads later
        thread.start()

    #terminate threads
    for thread in thread_list:
        thread.join()

    
    # for i in range(iterations):
    #     filtering.median_3x3(tmpA, tmpB, 0, 1)
    #     # swap direction of filtering
    #     tmpA, tmpB = tmpB, tmpA

    return tmpA

def image_processing_multithread(thread_idx, iterations, num_threads, event_array, tmpA, tmpB):
    '''A thread will go through image processing here.
    And we should make sure threads n, n-1, and n+1 have completed iteration i-1'''
    
    for iteration in range(iterations):
        # if iteration == 0 then, does not have iteration i-1, so pass
        if iteration > 0 and num_threads > 1:
            if thread_idx == 0 : #does not have n-1 thread
                event_array[iteration-1][thread_idx].wait()
                event_array[iteration-1][thread_idx + 1].wait()
            elif thread_idx == num_threads-1 : #does not have n+1 thread
                event_array[iteration-1][thread_idx - 1].wait()
                event_array[iteration-1][thread_idx].wait()
            else :
                event_array[iteration-1][thread_idx - 1].wait()
                event_array[iteration-1][thread_idx].wait()
                event_array[iteration-1][thread_idx + 1].wait()
        
        filtering.median_3x3(tmpA, tmpB, thread_idx, num_threads)
        
        # set this thread's event is ready
        event_array[iteration][thread_idx].set()

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
        new_image = py_median_3x3(input_image, 10, 4)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
