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
    # create a new memory location with image data for tmpA
    tmpA = image.copy()
    # create an empty structure for tmpB with same dimensions as tmpA
    tmpB = np.empty_like(tmpA)
    # initialize num_threads-sized list for threads
    threads = [None] * num_threads
    # initialize (num-threads * (iterations)-sized list for events
    events = [None] * num_threads * (iterations)

    # initialize events across all elements in list
    for threadidx in range(num_threads * (iterations)):
        # initialize each position with an event
        events[threadidx] = threading.Event()
    
    # reshape the events list to correspond to a 2D matrix
    events = np.reshape(events, [num_threads, iterations])

    # iterate for each thread
    for threadidx in range(num_threads):
        # initialize each thread with the appropriate arguments to the worker function
        threads[threadidx] = threading.Thread(target = worker, args = (threadidx, events, tmpA, tmpB, num_threads, iterations))
        # exit even when this thread is alive
        threads[threadidx].daemon = True
        # start the thread
        threads[threadidx].start()

    # collect and end the threads
    for t in threads:
        t.join()

    # return our result
    return tmpA


def worker(threadidx, events, tmpA, tmpB, num_threads, iterations):
    # initialize the list of adjacent threads
    adj_threads = []
    # set up adjacent threads (representing n-1 and n+1 threads) 
    for i in range(threadidx-1, threadidx+2):
        # exclude indices corresponding to edges
        if (i >= 0 and i < num_threads):
            adj_threads.append(i)
            
    # 10 interations
    for i in range(iterations):
        # skip event-checking if this is the first iteration
        if i != 0:
            # iterate only among adjacent threads (e.g., other threads are not touched)
            for j in adj_threads:
                # for these threads, block until the filtering/switching has occurred
                events[j, i-1].wait()

        # call median_3x3 Cython code with threadidx-multiple rows and offset num_threads
        filtering.median_3x3(tmpA, tmpB, threadidx, num_threads)
        # switch only the pertinent values of tmpA and tmpB
        tmpA, tmpB = tmpB, tmpA
        #for j in range(threadidx, len(tmpA), num_threads):
        #    tmpA[j], tmpB[j] = tmpB[j], tmpA[j]
        # event has occurred, so "release" the event
        events[threadidx, i].set()
        

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

    num_threads = range(1,9)
    thread_times = []
    for threads in num_threads:
        with Timer() as t:
            new_image = py_median_3x3(input_image, 10, threads)
        print("{} seconds for 10 filter passes with {} threads.".format(t.interval, threads))
        thread_times.append(t.interval)
        
    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    pylab.show()

    pylab.figure()
    pylab.plot(num_threads,thread_times)
    pylab.xlabel("Number of Threads")
    pylab.ylabel("Seconds")
    pylab.title("Time vs. # Threads")
    pylab.ylim([0,5])
    pylab.show()
    
