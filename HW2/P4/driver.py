import sys
import os.path
sys.path.append(os.path.join('..', 'util'))

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import numpy as np
import pylab
import pylab as plt
import seaborn as sns

import filtering
from timer import Timer
from threading import Thread


def py_median_3x3(image, iterations=10, num_threads=1):
    """Repeatedly filter with a 3x3 median"""
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)
    for i in xrange(iterations):
        threads = []
        # Fire up some threads
        for n in xrange(num_threads):
            thread = Thread(target=filtering.median_3x3, args=(tmpA, tmpB, n, num_threads))
            thread.start()
            threads += [thread]
        for t in threads:
            t.join()  # wait until done and kill threads
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA
    return tmpA


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

    #############################
    # Multithreading experiment #
    #############################

    threadcount = [1, 2, 4]
    execution_times = []
    for n in threadcount:
        with Timer() as t:
            new_image = py_median_3x3(input_image, 10, n)
        execution_times += [t.interval]

    f, ax = plt.subplots(1)
    # sns.set_style("darkgrid")
    plt.plot(threadcount, execution_times)
    plt.xlabel('N')
    plt.ylabel('Time (sec.)')
    plt.title('Execution time for 1, 2 and 4 threads')
    ax.set_ylim(0,)
    plt.show()
