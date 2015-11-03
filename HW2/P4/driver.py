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


def median_slice(tmpA, tmpB, i, iterations, semaphores):
    for _iter in range(iterations):
        #  number of threads = number of semaphore sets
        n_slices = len(semaphores)

        #  offset = current thread index
        start_row = i

        #  acquire semaphores from adjacent slices
        semaphores[i-1][1].acquire()
        semaphores[(i+1) % n_slices][0].acquire()

        filtering.median_3x3(tmpA, tmpB, start_row, n_slices)

        tmpA, tmpB = tmpB, tmpA

        #  release semaphores to adjacent slices
        semaphores[i][0].release()
        semaphores[i][1].release()

def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    semaphores = [(threading.Semaphore(), threading.Semaphore()) for t in range(num_threads)]
    threads = [
        threading.Thread(target=median_slice, args=(tmpA, tmpB, i, iterations, semaphores))
        for i in range(num_threads)
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

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

    thread_counts = [1, 2, 4, 6, 8, 16, 32, 64]
    for n_threads in thread_counts:
        with Timer() as t:
            new_image = py_median_3x3(input_image, 10, n_threads)
        print("{} seconds for 10 filter passes with {} threads.".format(t.interval, n_threads))

    # pylab.figure()
    # pylab.imshow(new_image[1200:1800, 3000:3500])
    # pylab.title('after - zoom')

    # pylab.show()
