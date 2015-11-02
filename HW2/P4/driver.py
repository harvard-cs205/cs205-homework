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
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )

def py_median_3x3(image, iterations=10, num_threads=4):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    threads = []
    events_matrix = [[threading.Event()] * num_threads for i in range(iterations)]
    for idx in range(num_threads):
        t = threading.Thread(target=worker, args=(idx, tmpA, tmpB, iterations, num_threads, events_matrix))
        threads.append(t)
        t.start()
    # for t in threads:
    #     t.join()
    # swap direction of filtering
    for t in threads:
        logging.debug('joining %s', t.getName())
        t.join()

    return tmpA

def worker(thn, tmpA, tmpB, iterations, num_threads, events_matrix):
    for i in range(iterations):
        if i == 0:
            filtering.median_3x3(tmpA, tmpB, thn, num_threads)
            events_matrix[i][thn].set()
        else:
            pre_thread = thn - 1 if thn > 0 else num_threads-1
            post_thread = thn + 1 if thn < num_threads-1 else 0

            while not (events_matrix[i - 1][pre_thread].is_set() and events_matrix[i - 1][post_thread].is_set() and events_matrix[i - 1][thn].is_set()):
                logging.debug('Waiting for adjacent threads ready...')
                events_matrix[i][thn].wait(1)

            filtering.median_3x3(tmpA, tmpB, thn, num_threads)

        events_matrix[i][thn].set()

        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA




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
        new_image = py_median_3x3(input_image, 10, 1)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
