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

def filter_component(in_image, out_image, events, num_iterations, num_threads):
    # Get the current thread id so we know what rows to operate on
    curr_thread_id = int(threading.currentThread().getName())

    #print 'Current thread is:', curr_thread_id

    # Now filter as many times as we desire
    for ii in range(num_iterations):
        # Make sure that the necessary computations are already completed
        # Clearly on the first iteration we should just go ahead
        # The modular arithmetic just wraps around for edge cases
        if ii > 0:
            events[curr_thread_id][ii - 1].wait()
            events[(curr_thread_id - 1) % num_threads][ii - 1].wait()
            events[(curr_thread_id + 1) % num_threads][ii - 1].wait()

        # Now that we've waited for everything, we can definitely go ahead
        #print 'About to filter on iteration', ii
        filtering.median_3x3(in_image, out_image, curr_thread_id, num_threads)
        #print 'After filter on iteration', ii

        # Swap the direction of filtering
        in_image, out_image = out_image, in_image

        # And let everybody else know that we finished up
        events[curr_thread_id][ii].set()



def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    # Create a matrix of events corresponding to (thread, iteration) pairs
    # Essentially, we will set event (t, i) when thread t has completed iteration i
    # Then when thread thread t' wants to try iteration i', we see if (t', i'-1), (t'-1, i-1), (t'+1, i-1) are done
    events  = [[threading.Event() for ii in range(iterations)] for jj in range(num_threads)]

    # Create our list of threads and give them all a name corresponding to what thread they are
    threads = [threading.Thread(name=str(jj), target=filter_component, args=(tmpA[::], tmpB[::], events, iterations, num_threads)) for jj in range(num_threads)]

    # Now run all the threads!
    for jj in range(num_threads):
        threads[jj].start()

#    for i in range(iterations):
        #filtering.median_3x3(tmpA, tmpB, 0, 1)
        ## swap direction of filtering
        #tmpA, tmpB = tmpB, tmpA

    # And now make sure that they all actually finished
    for jj in range(num_threads):
        threads[jj].join()

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
    #from_cython = py_median_3x3(input_image, 50, 4)
    #from_numpy = numpy_median(input_image, 50)
    #assert np.all(from_cython == from_numpy)

    with Timer() as t:
        new_image = py_median_3x3(input_image, 10, 4)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
