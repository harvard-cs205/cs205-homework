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

class Filter(threading.Thread):

    def __init__(self, imageA, imageB, iterations, thread_id, num_threads, semaphore_list):
        threading.Thread.__init__(self)
        self.imageA = imageA
        self.imageB = imageB
        self.iterations = iterations
        self.thread_id = thread_id
        self.num_threads = num_threads
        self.semaphore_list = semaphore_list

    def run(self):
        for i in range(self.iterations):
            #print("%d, %d, wait" % (self.thread_id, i))
            if i > 0:
                if self.thread_id > 0:
                    self.semaphore_list[self.thread_id - 1][i][0].acquire()
                if self.thread_id + 1 < self.num_threads:
                    self.semaphore_list[self.thread_id + 1][i][1].acquire()

            filtering.median_3x3(self.imageA, self.imageB, self.thread_id, self.num_threads)

            #print("%d, %d, release" % (self.thread_id, i))
            if i + 1 < self.iterations:
                self.semaphore_list[self.thread_id][i + 1][0].release()
                self.semaphore_list[self.thread_id][i + 1][1].release()

            self.imageB, self.imageA = self.imageA, self.imageB


def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    semaphore_list = []
    for thread_id in range(num_threads):
        thread_semaphore_list = []
        for i in range(iterations):
            s = []
            for j in range(2):
                s.append(threading.Semaphore(0))
            thread_semaphore_list.append(s)
        semaphore_list.append(thread_semaphore_list)

    fl_list = []
    for thread_id in range(num_threads):
        fl = Filter(tmpA, tmpB, iterations, thread_id, num_threads, semaphore_list)
        fl.start()
        fl_list.append(fl)
    for fl in fl_list:
        fl.join()

    """
    for i in range(iterations):
        filtering.median_3x3(tmpA, tmpB, 0, 1)
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA
    """

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
        new_image = py_median_3x3(input_image, 10, 5)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
