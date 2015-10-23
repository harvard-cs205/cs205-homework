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

class Row_Handler():
    def __init__(self, row_num, tmpA, tmpB):
        self.i = 0

        self.row_num = row_num

        self.above_handler = None
        self.below_handler = None

        # Run when you are ready to go!
        self.row_lock = threading.Lock()

        self.tmpA = tmpA
        self.tmpB = tmpB


    def go(self):
        if (self.i == self.above_handler.i) and (self.i == self.below_handler.i):
            self.above_handler.row_lock.acquire()
            self.row_lock.acquire()
            self.below_handler.acquire()

            # Do stuff
            filtering.median_3x3_row(self.tmpA, self.tmpB, self.row_num)
            # Swap tmpA and tmpB; doesn't recreate arrays, just swaps pointers!
            potatoA = self.tmpA
            potatoB = self.tmpB

            self.tmpA = potatoB
            self.tmpB = potatoA

            self.above_handler.row_lock.release()
            self.row_lock.release()
            self.below_handler.release()



def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    for i in range(iterations):
        filtering.median_3x3(tmpA, tmpB, 0, 1)
        # swap direction of filtering
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
