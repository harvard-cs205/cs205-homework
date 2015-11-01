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

def median_3x3_worker(A, B, offset, step, iterations, sems) : 
    for i in range(iterations):
        lower_sem = sems[(offset - 1) % step][1]
        upper_sem = sems[(offset + 1) % step][0]

        (self_sem_1, self_sem_2) = sems[offset]
        lower_sem.acquire()
        upper_sem.acquire()
       
        filtering.median_3x3(A, B, offset, step)

        # swap direction of filtering
        A, B = B, A

        self_sem_1.release()
        self_sem_2.release()


def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    sems = []
    for i in range(num_threads) :
        sems.append((threading.Semaphore(1), threading.Semaphore(1)))

    threads = []
    for i in range(num_threads) :
        t = threading.Thread(target=median_3x3_worker, name=str(i), args=(tmpA, tmpB, i, num_threads, iterations, sems))
        t.start()
        threads.append(t)

    for t in threads :
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

    with Timer() as t:
        new_image = py_median_3x3(input_image, 10, 4)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
