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
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    for i in range(iterations):
        filtering.median_3x3(tmpA, tmpB, 0, 1)
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA

    return tmpA

# the multithread version
def py_median_3x3_multi(image, iterations=10, num_threads=4):
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)
    
    def py_median_3x3_single(tmpA, tmpB, idx):
        filtering.median_3x3(tmpA, tmpB, idx, num_threads)
    
    for i in range(iterations):
        thlist = []
        for idx in range(num_threads):
            # create threads
            th = threading.Thread(target = py_median_3x3_single,
                                  args = (tmpA, tmpB, idx))
            th.daemon = True
            thlist.append(th)
            # start to run
            th.start()

        # join all threads to main when finished
        for th in thlist:
            th.join()

        tmpA, tmpB = tmpB, tmpA

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
    from_cython = py_median_3x3_multi(input_image, 2, 5)
    from_numpy = numpy_median(input_image, 2)
    assert np.all(from_cython == from_numpy)

    with Timer() as t:
        new_image = py_median_3x3(input_image, 10, 8)

    print("{} seconds for serial 10 filter passes.".format(t.interval))
    pylab.show()

    for num_threads in [1, 2, 4]:
        with Timer() as t:
            new_image = py_median_3x3_multi(input_image, 10, num_threads)
        print 'num_threads =', num_threads
        print("{} seconds for multithread 10 filter passes.".format(t.interval))

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')
    pylab.show()
