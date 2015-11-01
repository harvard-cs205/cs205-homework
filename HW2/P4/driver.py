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
from multiprocessing.pool import ThreadPool

def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median and multiple threads '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)
    events = []
    threads = []

    # events[] has an event for each thread i finishing iteration n. It looks like:
    # [thr0 iter1, thr0 iter2, ..., thr0 iter k, thr1 iter1, ..., thrN iterK]
    # This way, every thread can see when a particular thread has finished a
    # particular iteration.
    #
    # There are more details in P4.txt

    def process_image(threadnum, tmpA, tmpB, offset=num_threads):
        ''' do filtering for the threadnum-th thread '''
        for iter in range(iterations):
            # We don't have to check for this thread finishing iteration i-1 because
            # it does each of its iterations in sequence.
            if iter > 0:
                if threadnum != 0:
                    events[(threadnum-1)*iterations+iter-1].wait()
                if threadnum != num_threads-1:
                    events[(threadnum+1)*iterations+iter-1].wait()
            filtering.median_3x3(tmpA, tmpB, threadnum, offset)
            tmpA, tmpB = tmpB, tmpA
            events[threadnum*iterations+iter].set()

    # Start the threads and initialize event variables
    for i in range(num_threads):
        threads.append(threading.Thread(target=process_image, name='Thread'+str(i+1),
                         args=(i, tmpA, tmpB)))
        for iter in range(iterations):
            events.append(threading.Event())
        threads[i].start()

    # Wait for threads to finish
    for thr in threads:
        thr.join()

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
