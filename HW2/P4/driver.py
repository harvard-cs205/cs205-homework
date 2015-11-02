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
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)
    num_lines = tmpA.shape[0]
    #list of lists where lines_by_tid[tid] = list of lines that tid is responsible
    # e.g. tid == 1 ==> lines_by_tid[1] = [1 + num_threads * 0, 1 + num_threads * 1, 1 + num_threads * 2,...]
    '''
    lines_by_tid = []
    for tid in range(num_threads):
        #tid is thread_id
        tid_lines = []
        # this will fail if num_threads > # of lines
        for x in range(num_lines/num_threads):
            tid_lines.append(tid + num_threads * x)
        lines_by_tid.append(tid_lines)
    #if # lines is not divisible by num threads, distribute remaining lines
    x = num_lines/num_threads
    for tid in range(num_lines%num_threads):
        lines_by_tid[tid].append(tid + num_threads * x)
    '''  
    #print threading.active_count()
    for i in range(iterations):
        threads = []
        for tid in range(num_threads):
            #offset = tid, step = num_threads
            t = threading.Thread(target=filtering.median_3x3,args=(tmpA, tmpB, tid, num_threads))
            threads.append(t)
            t.start()
        for t in threads:
            #print t.is_alive()
            t.join()
        #print threading.active_count()
        #assert threading.active_count() == 1
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA

    return tmpA


def py_median_3x3_orig(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    for i in range(iterations):
        filtering.median_3x3(tmpA, tmpB, 0, 1)
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
        new_image = py_median_3x3(input_image, 10, 4)
        #new_image = py_median_3x3(input_image, 10, 8)
    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
