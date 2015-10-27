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

def median3x3_worker(tmpA, tmpB, offset, step, iterations, event_start_iterate, event_list, threadidx):
    #median_3x3 takes in step and offset
    #step is the number of threads
    #offset is the initial thread Number

    for i in range(iterations):

        #all threads wait for next iteration to start. controlled by main thread
        #print "thread " + str(threadidx) + " waiting to start"
        event_start_iterate.wait()



        if i%2 == 0:
            filtering.median_3x3(tmpA, tmpB, offset, step)    
        else:
            filtering.median_3x3(tmpB, tmpA, offset, step)   

        #set the start iterate to false so that the threads will wait for the next iteration to start
        event_start_iterate.clear()

        #print "thread " + str(threadidx) + " ended"

        #tell the main thread that this process has ended
        event_list[threadidx].set()


def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    event_list = []

    #swap is done here 
    for x in range(num_threads):
        event_list.append(threading.Event())    

    event_start_iterate = threading.Event()

    for threadidx in range(num_threads):
        th = threading.Thread(target=median3x3_worker,
                                  args=(tmpA, tmpB, threadidx, num_threads, iterations,
                                         event_start_iterate, event_list, threadidx))

        th.start()

    for i in range(iterations):
        for y in range(num_threads):
            event_list[y].clear()
        #start the next iteration
        event_start_iterate.set()
        #print "Starting all threads"
        for y in range(num_threads):
            event_list[y].wait()

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
        new_image = py_median_3x3(input_image, 10, 4)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
