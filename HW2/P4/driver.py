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


# thread class
class workerThread(threading.Thread):
    def __init__(self, threadID, offset, N, tmpA, tmpB):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.offset = offset
        self.N = N
        self.tmpA = tmpA
        self.tmpB = tmpB

    def run(self):
        # perform filtering
        filtering.median_3x3(self.tmpA, self.tmpB, self.offset, self.N)

def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)


    for i in range(iterations):
        # one thread version
        #filtering.median_3x3(tmpA, tmpB, 0, 1)

        # create num_threads threads
        threads = []
        for i in range(num_threads):
            thread = workerThread(i, i, num_threads, tmpA, tmpB)
            threads.append(thread)

        # using n threads, start all of them
        for t in threads:
            t.start()

        # wait till are finished, before going to the next iteration
        for t in threads:
            t.join()

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
