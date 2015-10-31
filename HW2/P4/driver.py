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
    # repeatedly filter with a 3x3 median
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    # Each thread controls every Nth line in image
    threads = [None] * num_threads
    # Each event corresponds to a particular thread and particular iteration - represents whether that thread
    # is done with that iteration
    events = [[threading.Event() for i in range(iterations)] for j in range(num_threads)]

    for thread_id in range(num_threads):
        threads[thread_id] = threading.Thread(target=filter_image, args=(num_threads, thread_id, tmpA, tmpB, events, iterations))
        threads[thread_id].start()

    # Terminate all threads
    for thread in threads:
        thread.join()

    # Return fully filtered image
    return tmpA

def filter_image(num_threads, thread_id, tmpA, tmpB, events, iterations=10):
    for itr in range(iterations):
        # Waits on itr-1 of thread_id-1 and thread_id+1 to finish itr-1 before executing current itr
        if num_threads > 1 and itr > 0:
            events[(thread_id-1)%num_threads][itr-1].wait()
            events[(thread_id+1)%num_threads][itr-1].wait()
        filtering.median_3x3(tmpA, tmpB, thread_id, num_threads)
        # Set event flag to GO
        events[thread_id][itr].set()
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
