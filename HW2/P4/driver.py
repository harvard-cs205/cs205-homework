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

# A function that can be called by parallel threads to filter specific rows of an image, a certain number of times
def filtering_thread(tmpA, tmpB, thread_id, num_threads, event_dict, iterations):
    for i in range(iterations):
        # For N threads, the nth thread should process every 'num_threads' row, starting with row 'n'
        filtering.median_3x3(tmpA, tmpB, thread_id, num_threads)
        
        # Set the Event for this iteration and thread_id combo, so 
        # other threads that are blocked by this Event know it is 
        # ok to continue
        event_dict[(i, thread_id)].set()
        
        # Swap direction of filtering
        tmpA, tmpB = tmpB, tmpA
        
        # First iteration doesn't need to wait on anything, while 
        # subsequent iterations should wait on the adjacent threads, 
        # which is why we do this wait at the end of the iteration 
        # looping
        
        if num_threads == 1:
            # If we only have 1 thread, then just continue
            continue
        elif num_threads == 2:
            # If we have exactly 2 threads, just wait on the other one
            if thread_id == 0:
                event_dict[(i, 1)].wait()
            if thread_id == 1:
                event_dict[(i, 0)].wait()
        else:
            # More than 2 threads is a bit more complicated:
            if thread_id == 0:
                # The 0th thread should wait on the 1st thread, and the last thread
                event_dict[(i, thread_id + 1)].wait()
                event_dict[(i, num_threads - 1)].wait()
            elif thread_id == (num_threads - 1):
                # The last thread should wait on the penultimate thread, and the 0th thread
                event_dict[(i, thread_id -1)].wait()
                event_dict[(i, 0)].wait()
            else:
                # Middle threads should wait on their neighbors
                event_dict[(i, thread_id - 1)].wait()
                event_dict[(i, thread_id + 1)].wait()

            
def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)
    
    # A dictionary to store the threading.Event objects for 
    # specific iterations and threads, in the format:
    #     {(<ITERATION_ID>, <THREAD_ID>): <EVENT>, ... }
    event_dict = {}
    
    # Create Events for every iteration-thread and store them in a dict
    for i in range(iterations):
        for t_id in range(num_threads):
            e = threading.Event()
            event_dict[(i,t_id)] = e

    # An array to hold the threads
    threads = []
    
    # Create num_threads threads and have them run filtering_thread
    for thread_id in range(num_threads):
        a_thread = threading.Thread(target=filtering_thread, args=(tmpA, tmpB, thread_id, num_threads, event_dict, iterations))
        a_thread.start()
        threads.append(a_thread)

    # Wait for all of the threads to complete before returning the filtered image
    for a_thread in threads:
        a_thread.join()
    
    # Return the filtered image
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
    #assert np.all(from_cython != input_image)
    from_numpy = numpy_median(input_image, 2)
    assert np.all(from_cython == from_numpy)

    with Timer() as t:
        new_image = py_median_3x3(input_image, 10, 4)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
