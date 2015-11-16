import sys
import os.path
sys.path.append(os.path.join('..', 'util'))

import set_compiler
set_compiler.install()

import pyximport
pyximport.install()

import matplotlib.pyplot as plt

import numpy as np
import pylab

import filtering
from timer import Timer
import threading


def waiter(conds, thread, iteration, num_threads):
    # Make sure you have waited for necessary pre-processed steps.
    if iteration == 0:
        return
    if thread - 1 >= 0:
        conds[iteration-1][thread-1].wait()
    if thread + 1 < num_threads:
        conds[iteration-1][thread+1].wait()
    conds[iteration-1][thread].wait()


def thread_worker(conds, num_iterations, thread_id, tmpA, tmpB, num_threads):
    for i in range(num_iterations):
        waiter(conds, thread_id, i, num_threads)
        filtering.median_3x3(tmpA, tmpB, thread_id, num_threads)
        tmpA, tmpB = tmpB, tmpA
        conds[i][thread_id].set()

def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    # There are iterations rows, and num_threads columns
    # Create an Event() for each of those which will be set() once that (n, i) pair has been done.
    conditions = [[threading.Event() for j in range(num_threads)] for i in range(iterations)]
    # Instantiate the threads.
    threads = [threading.Thread(target=thread_worker, args=(conditions, iterations, t, tmpA, tmpB, num_threads)) for t in range(num_threads)]
    # Start the threads
    for t in threads:
        t.start()
    # Wait for all of the threads to finish.
    for t in threads:
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

    x_threads = []
    y_time = []
    for nt in [1,2,4]:
        for _ in range(5):
            with Timer() as t:
                new_image = py_median_3x3(input_image, 10, nt)
            print("Threads: {}. Time: {} seconds for 10 filter passes.".format(nt, t.interval))
            x_threads.append(nt)
            y_time.append(t.interval)
    plt.figure()
    plt.scatter(x_threads, y_time)
    plt.xlabel("Number of Threads")
    plt.ylabel("Time")
    plt.title("Performance of Algorithm by Number of Threads")
    plt.show()

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')
    pylab.show()
