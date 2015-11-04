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

mtx_rem = threading.Lock()
rem = 0
fin = threading.Event()
st = threading.Event()

def runThread(tmpA, tmpB, ID, nThreads, iter):
    global rem
    for i in xrange(iter):
        st.wait() # Wait for the signal
        for r in xrange(ID, len(tmpA), nThreads):
            uRow = max(0,r-1)
            dRow = min(r+2, tmpA.shape[0])
            offset = int(r > 0)
            filtering.median_3x3(tmpA[uRow:dRow, :], tmpB[uRow:dRow, :], offset, 2)

        mtx_rem.acquire()
        rem -= 1
        if rem == 0:
            fin.set()
            fin.clear()
        mtx_rem.release()

        tmpA, tmpB = tmpB, tmpA

def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    global rem, event
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)
    event = threading.Event()
    threads = [threading.Thread(target=runThread, args=(tmpA, tmpB, i, num_threads, iterations)) for i in xrange(
        num_threads)]

    for t in threads:
        t.start()

    for i in range(iterations):
        mtx_rem.acquire()
        rem = num_threads
        mtx_rem.release()

        st.set()
        st.clear()

        fin.wait()

        tmpA, tmpB = tmpB, tmpA
        # for r in xrange(image.shape[0]):
        #     uRow = max(0,r-1)
        #     dRow = min(r+2,image.shape[0])
        #     offset = int(r > 0)
        #     filtering.median_3x3(tmpA[uRow:dRow, :], tmpB[uRow:dRow, :], offset, 2)
        # swap direction of filtering


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

    # res = (from_cython == from_numpy)
    #
    # for i in xrange( len(from_cython) ):
    #     for j in xrange( len(from_cython) ):
    #         if not res[i, j]:
    #             print 'No match: at ', i, ' ' , j, ' with ', res[i,j]
    #             print from_cython[i, j], ' vs ', from_numpy[i, j]

    assert np.all(from_cython == from_numpy)

    with Timer() as t:
        new_image = py_median_3x3(input_image, 10, 8)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
