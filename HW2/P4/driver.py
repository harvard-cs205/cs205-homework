# Note: includes all my comments from reviewing the skeleton code

# Collaborated with Kendrick Lo

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

# Original code without threading
def py_median_3x3_serial(image, iterations):
    ''' repeatedly filter with a 3x3 median '''
    
    # Create a copy of the image being processed
    tmpA = image.copy()
    
    # Create a blank buffer with same dimensions as tmpA
    tmpB = np.empty_like(tmpA)

    # Repeatedly apply median filtering
    for i in range(iterations):

        # Applies median filtering
        filtering.median_3x3(tmpA, tmpB, 0, 1)
        
        # Swap direction of filtering
        tmpA, tmpB = tmpB, tmpA

    # Return filtered image
    return tmpA

# Helper function - identifies dependent threads for the asynchronous case
def dependent_threads(thread_id, num_threads):

    # No dependencies
    if num_threads == 1:
        return []

    # Dependent on all other threads
    elif num_threads < 4:
        return [k for k in range(num_threads) if k != thread_id]

    # Dependent on threads to the 'left' and 'right'
    else:
        if thread_id - 1 < 0:
            result = [num_threads-1]
        else:
            result = [thread_id - 1]
        if thread_id + 1 >= num_threads:
            result.append(0)
        else:
            result.append(thread_id + 1)
        result.sort()
        return result

# Worker function (synchronous)
def wrap_median_3x3_synch(imgA, imgB, thread_id, num_threads, iterations, events):
    
    # Loop through iterations for a given thread
    for i in range(iterations):

        # Clear event flag for current thread-iteration pair - i.e. is not complete
        events[thread_id][i].clear()

        # Apply median filtering
        filtering.median_3x3(imgA, imgB, thread_id, num_threads)

        # Swap direction of filtering
        imgA, imgB = imgB, imgA

        # Set event flag for current thread-iteration pair - i.e. is complete
        events[thread_id][i].set()

        # Wait for all other threads to complete the current iteration
        keyThreads = [k for k in range(num_threads) if k != thread_id]
        for t in keyThreads:
            events[t][i].wait()

# Worker function (asynchronous)
def wrap_median_3x3_asynch(imgA, imgB, thread_id, num_threads, iterations, events):
    
    # Loop through iterations for a given thread
    for i in range(iterations):

        # Clear event flag for current thread-iteration pair - i.e. is not complete
        events[thread_id][i].clear()

        # Apply median filtering
        filtering.median_3x3(imgA, imgB, thread_id, num_threads)

        # Swap direction of filtering after all threads are done
        imgA, imgB = imgB, imgA

        # Set event flag for current thread-iteration pair - i.e. is complete
        events[thread_id][i].set()

        # Only wait for threads with dependencies to complete the current iteration
        keyThreads = dependent_threads(thread_id, num_threads)
        for t in keyThreads:
            events[t][i].wait()
 
# Python code with threading
def py_median_3x3(image, thread_func, iterations, num_threads):
    ''' repeatedly filter with a 3x3 median '''
    
    # Create a copy of the image being processed
    tmpA = image.copy()
    
    # Create a blank buffer with same dimensions as tmpA
    tmpB = np.empty_like(tmpA)

    # Create an array of events - all flags are cleared at initialization
    # Events for thread-iteration pairs are called by events[thread_id][iteration]
    events = [[threading.Event()  for x in range(iterations)] for y in range(num_threads)] 

    # Create an array of threads (filled in below)
    threads = [None for t_create in range(num_threads)]

    # Create the necessary number of threads and pass the synchronous/asynchronous worker function
    for t_start in range(num_threads):
        threads[t_start] = threading.Thread(target=thread_func, args=(tmpA, tmpB, t_start, num_threads, iterations, events))
        threads[t_start].start()

    # Wait until all threads terminate
    for t_end in range(num_threads):
        threads[t_end].join()

    # Clean up - delete thread and events arrays
    del threads, events

    # Return filtered image
    return tmpA

# Used to check correct functioning of threaded code
def numpy_median(image, iterations):
    ''' filter using numpy '''

    # Repeatedly apply median filtering
    for i in range(iterations):
        
        # Obtain 3x3 block of pixels surrounding each pixel
        padded = np.pad(image, 1, mode='edge')

        # Stack arrays in sequence depth-wise (prepares for processing)
        stacked = np.dstack((padded[:-2,  :-2], padded[:-2,  1:-1], padded[:-2,  2:],
                             padded[1:-1, :-2], padded[1:-1, 1:-1], padded[1:-1, 2:],
                             padded[2:,   :-2], padded[2:,   1:-1], padded[2:,   2:]))
        
        # Return median of stacked array (i.e. of 3x3 block around each pixel)
        image = np.median(stacked, axis=2)

    return image

if __name__ == '__main__':

    ########################################
    # Development & testing
    ########################################

    # # Load image
    # input_image = np.load('image.npz')['image'].astype(np.float32)

    # # Set default colormap to gray and apply to image
    # pylab.gray()

    # # Check that implementations are correct

    # noThreads = 4
    # noIterations = 10

    # from_numpy = numpy_median(input_image, iterations=noIterations)

    # # Synchronous implementation
    # from_cython_synch = py_median_3x3(input_image, wrap_median_3x3_synch, iterations=noIterations, num_threads=noThreads)
    # assert np.all(from_cython_synch == from_numpy)

    # # Asynchronous implementation
    # from_cython_asynch = py_median_3x3(input_image, wrap_median_3x3_asynch, iterations=noIterations, num_threads=noThreads)
    # assert np.all(from_cython_asynch == from_numpy)

    ########################################
    # Live version: apply filter & display image
    ########################################

    # Load image
    input_image = np.load('image.npz')['image'].astype(np.float32)

    # Set default colormap to gray and apply to image
    pylab.gray()

    # Display original (full-size) image
    pylab.imshow(input_image)
    pylab.title('original image')

    # Display selected portion of image
    pylab.figure()
    pylab.imshow(input_image[1200:1800, 3000:3500])
    pylab.title('before - zoom')

    # Time median filtering
    with Timer() as t:
        new_image = py_median_3x3(input_image, wrap_median_3x3_asynch, iterations=10, num_threads=8)

    # Display selected portion of updated image
    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    # Print runtime
    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()