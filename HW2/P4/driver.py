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
    # repeatedly filter with a 3x3 median 
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    for i in range(iterations):
        #filtering.median_3x3(tmpA, tmpB, 0, 1)
        threads = []
        for j in range(num_threads):
            cur = threading.Thread(target=filtering.median_3x3, args=(tmpA, tmpB, j, num_threads))
            threads.append(cur)
            cur.start()

        # wait until all threads are finished
        for thread in threads:
            thread.join()

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
    input_image = input_image[::2, ::2].copy()
    #input_image = input_image[:1100, :4000].copy()
    print input_image.shape

    #pylab.gray()

    #pylab.imshow(input_image)
    #pylab.title('original image')

    #pylab.figure()
    #pylab.imshow(input_image[1200:1800, 3000:3500])
    #pylab.imshow(input_image[600:900, 1500:1750])
    #pylab.title('before - zoom')

    # verify correctness
    for num_threads in [1,2,4]:
        from_cython = py_median_3x3(input_image, 10, num_threads)
        from_numpy = numpy_median(input_image, 10)
        assert np.all(from_cython == from_numpy)

        with Timer() as t:
            new_image = py_median_3x3(input_image, 10, num_threads)
        '''
        pylab.figure()
        pylab.imshow(new_image[600:900, 1500:1750])
        pylab.title('after - zoom')
        '''

        print("number of threads={}, {} seconds for 10 filter passes.".format(num_threads, t.interval))
        #pylab.show()
