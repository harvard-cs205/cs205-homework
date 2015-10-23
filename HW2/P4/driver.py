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

from or_event import OrEvent
from threadpool import ThreadPool

class Row_Handler():
    def __init__(self, row_num, tmpA, tmpB, num_iterations=10):
        self.i = 0

        self.row_num = row_num

        self.above_handler = None
        self.below_handler = None

        # Run when you are ready to go!
        self.row_lock = threading.Lock()
        self.i_updated = threading.Event()

        self.tmpA = tmpA
        self.tmpB = tmpB
        self.num_iterations = num_iterations


    def go(self):
        go_cond_1 = True
        if self.above_handler is not None:
            go_cond_1 = (self.i == self.above_handler.i)
        go_cond_2 = True
        if self.below_handler is not None:
            go_cond_2 = (self.i == self.below_handler.i)
        if go_cond_1 and go_cond_2:
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

            self.i += 1

            self.above_handler.row_lock.release()
            self.row_lock.release()
            self.below_handler.release()

            # Give other threads a chance to grab locks and update
            self.i_updated.set()
            self.i_updated.clear()
        elif self.i != self.num_iterations: # you are not done yet!
            self.wait()

    def wait(self):
        """Wait until one of your neighbors updates their iteration"""
        OrEvent(self.above_handler.i_updated, self.below_handler.i_updated).wait()
        self.go()

    def done(self):
        print 'Done!'

def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    # Create all row_handlers
    handler_list = []
    for cur_row in range(image.shape[0]):
        handler_list.append(Row_Handler(cur_row, tmpA, tmpB, num_iterations=iterations))

    # Add neighbors
    for i in range(len(handler_list)):
        if i == 0:
            handler_list[i].below_handler =handler_list[i + 1]
        elif i == len(handler_list) - 1:
            handler_list[i]

    # The handlers must each be updated ten times...if we had a thread for each handler we would be all set...
    # we can probably subdivide the jobs up.

    pool = ThreadPool(num_threads)
    for handler in handler_list:
        pool.add_task(handler.go())

    pool.wait_completion()

    return handler_list[0].tmpA

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
