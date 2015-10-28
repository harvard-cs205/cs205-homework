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
from threading import Thread


def py_median_3x3(image, iterations=10, num_threads=1):
    """ repeatedly filter with a 3x3 median """
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    width = tmpA.shape[0]
    height = tmpA.shape[1]

    for i in xrange(iterations):
        threads = []

        # Loop over rows
        for j in xrange(0, image.shape[0], num_threads):

            # Loop over elements
            for k in xrange(image.shape[1]):
                out = [0] * num_threads

                # Tell every thread what to do in this row
                for n in xrange(num_threads):
                    if k+n+1 <= width and j+1 <= height and k+n-1 >= 0 and j-1 >= 0:
                        in_arg = tmpA[k+n-1:k+n+2, j-1:j+2]
                        out[n] = np.empty_like(in_arg)
                        t = Thread(name='{}'.format(n), target=filtering.median_3x3, args=(in_arg, out[n], 0, 1))
                        t.start()
                        threads += [t]
                map(lambda t: t.join(), threads)

                # Save result for this iteration in tmpB
                for n in xrange(len(out)):
                    if k+n+1 <= width and j+1 <= height and k+n-1 >= 0 and j-1 >= 0:
                        # print out[n]
                        tmpB[k+n, j] = out[n][1, 1]

        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA

    return tmpA


def py_median_3x32(image, iterations=10, num_threads=1):
    """Repeatedly filter with a 3x3 median"""
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)
    for i in xrange(iterations):
        threads = []
        # Fire up some threads
        for n in xrange(num_threads):
            thread = Thread(target=filter_for_thread, args=(n, num_threads, tmpA, tmpB))
            thread.start()
            threads += [thread]
        for t in threads:
            t.join()  # wait until done before going to new iteration
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA
    return tmpA


def filter_for_thread(n, num_threads, in_img, out_img):
    """Does the actual filtering"""
    for j in xrange(n, in_img.shape[1], num_threads):
        for i in xrange(0, in_img.shape[0]):
            p = take_pixel(i, j, in_img)
            out_p = take_pixel(i, j, out_img)
            if p is not None:
                filtering.median_3x3(p, out_p, 0, 1)


def take_pixel(row, column, source):
    """Returns a 3x3 pixels field"""
    width = source.shape[0]
    height = source.shape[1]
    if row + 1 <= width and column + 1 <= height and row -1 >= 0 and column - 1 >= 0:
        return source[row - 1: row + 2, column - 1: column + 2]


def numpy_median(image, iterations=10):
    """ filter using numpy """
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
    from_cython = py_median_3x32(input_image, 2, 5)
    new_image = from_cython
    from_numpy = numpy_median(input_image, 2)
    # assert np.all(from_cython == from_numpy)
    #
    # with Timer() as t:
    #     new_image = py_median_3x32(input_image, 10, 8)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
