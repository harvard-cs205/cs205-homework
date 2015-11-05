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

#Original:
"""
def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    for i in range(iterations):
        filtering.median_3x3(tmpA, tmpB, 0, 1)
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA

    return tmpA
"""

# Each thread is sent here
def worker(tmpA, tmpB, events, iterations, thread_id, num_threads):
    
    for i in range(iterations):
        
        filtering.median_3x3(tmpA, tmpB, thread_id, num_threads)
        
        # Let all threads that might be waiting for this iteration on this thread id know that 
        #    they can stop waiting and proceed.
        events[i, thread_id].set()

        # For single-threaded case, no other threads need to wait, so just move on.  
        if num_threads==1: 
            pass 

        # For two-threaded case, each thread needs to wait for the other from the same iteration.
        elif num_threads==2: 
            if thread_id == 0: events[i,1].wait()
            else: events[i,0].wait() 
        
        # For more than two threads, determine which other threads it needs to wait for before
        #    proceeding.  These are the threads that contain the rows immediately above and below, 
        #    but have to be defined uniquely for each case.  
	else:
            # For thread zero, wait for the thread containing the next rows down and the final rows.
            # These are the rows immediately above and below this thread's rows.  
            if thread_id==0: 
                events[i,1].wait()
                events[i,num_threads-1].wait()
            # For the highest number thread, wait for thread zero and the thread above it.  
            elif thread_id==num_threads-1:
                events[i,0].wait()
                events[i,thread_id-1].wait()
            # For middle threads, simply wait for the threads containing the rows above and below.  
            else:
                events[i,thread_id-1].wait()
                events[i,thread_id+1].wait()
       
        # Swap before the next iteration
        tmpA, tmpB = tmpB, tmpA
    
def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)
    
    # Empty lists to fill with threads and events below.  
    threads = []
    events = []
    
    # Create an array of events for each thread and iteration combo
    # These are used to signal when a thread can move forward because computations it relies on have completed.  
    for iteration in range(iterations*num_threads):
        events.append(threading.Event())
    # Reshape the array to be easy to call by iteration and thread id
    events = np.array(events).reshape([iterations,num_threads])

    # Create and start the number of threads specified
    for thread_id in range(num_threads):
        t = threading.Thread(target=worker, args=(tmpA, tmpB, events, iterations, thread_id,  num_threads))
        threads.append(t)
        t.start()
    
    # Wait until the threads terminate to return the updated array
    for t in range(num_threads):
        threads[t].join()

    # Once the threads are complete, return the updated array
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
