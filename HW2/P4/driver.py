import sys
import os.path
sys.path.append(os.path.join('..', 'util'))

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import numpy as np
import matplotlib
matplotlib.use('qt4agg')
import pylab

import filtering
from timer import Timer
import threading

class Worker(threading.Thread):

    def __init__(self, offset, step, tmpA, tmpB):
        # Initialize the super class
        threading.Thread.__init__(self)
        self.offset = offset
        self.step = step

        self.tmpA = tmpA
        self.tmpB = tmpB

    def run(self):
        filtering.median_3x3(self.tmpA, self.tmpB, self.offset, self.step)

def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    print 'Num threads:' , num_threads

    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    # Create numthreads for each iteration...they only run once.
    for i in range(iterations):
        thread_list = []
        for i in range(num_threads):
            thread_list.append(Worker(i, num_threads, tmpA, tmpB))

        map(lambda x: x.start(), thread_list)
        map(lambda x: x.join(), thread_list)

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
    num_threads_to_use = 4

    print 'Checking solution...'
    # Check with the appropriate number of threads
    from_cython = py_median_3x3(input_image, 10, num_threads_to_use)
    from_numpy = numpy_median(input_image, 10)
    assert np.all(from_cython == from_numpy)
    print 'Check passed! Now running timed code...'

    time_list = []
    for i in range(10):
        with Timer() as t:
            new_image = py_median_3x3(input_image, 10, num_threads_to_use)
        time_list.append(t.interval)

    time_list = np.array(time_list)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes averaged over 10 runs.".format(time_list.mean()))
    pylab.show()
