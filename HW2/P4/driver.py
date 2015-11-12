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

def py_median_3x3(image, iterations=10, num_threads=4):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    # Create an event that represents each call to filter_image (one event for each iteration in each thread)
    events = [[threading.Event() for i in xrange(iterations)] for j in xrange(num_threads)]

    # Generate list of threads and start them
    threads = []
    for i in xrange(num_threads):
        threads.append(threading.Thread(target=filter_image, args=(events, i, num_threads, tmpA, tmpB, iterations)))
        threads[i].start()

    # Join threads
    for thread in threads:
        thread.join()

    return tmpA

# Function called by each thread to perform filtering
def filter_image(events, thread_idx, num_threads, tmpA, tmpB, iterations):
    for i in xrange(iterations):
        # Ignore for i = 0 because we only wait on neighboring threads to complete i-1 iterations for i > 0
        # Ignore for num_threads <= 1 because no neighboring threads exist
        if num_threads > 1 and i > 0:
            # Need to wait for neighboring threads to complete i-1 iterations
            events[(thread_idx-1)%num_threads][i-1].wait()
            events[(thread_idx+1)%num_threads][i-1].wait()
        # Perform filtering once appropriate
        filtering.median_3x3(tmpA, tmpB, thread_idx, num_threads)

        # Set flag for ith event of current thread
        events[thread_idx][i].set()

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
        new_image = py_median_3x3(input_image, 10, 8)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
