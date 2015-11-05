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

def compute(threadID, num_threads, tmpA, tmpB, events, iterations = 10):
    for i in range(iterations):
        if i > 0:
            # wait for prev iteration of nearby threads; check boundaries
            if threadID > 0:
                events[threadID - 1][i-1].wait()
            if threadID < num_threads - 1:
                events[threadID + 1][i-1].wait()

        # compute
        filtering.median_3x3(tmpA, tmpB, threadID, num_threads)
        events[threadID][i].set()
        tmpA, tmpB = tmpB, tmpA

def py_median_3x3(image, iterations=10, num_threads=4):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    # make an event for a thread reaching a certain iteration
    events = []
    for i in range(num_threads):
        events.append([threading.Event() for _ in range(iterations)])
    
    # make threads compute
    threads = []
    for thread in range(num_threads):
        x = threading.Thread(target=compute,args=(thread,num_threads,tmpA,tmpB,events,iterations))
        x.start()
        threads.append(x)

    # end threads
    for y in threads:
        y.join()
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

    # scale down for my VM
    input_image = input_image[::2, ::2].copy()

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
