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
    events = []
    for i in range(num_threads):
        iter_list = []
        for j in range(iterations):
            iter_list.append(threading.Event())
        events.append(iter_list)

    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)
    threads=[]
    for i in range(num_threads):
        t=threading.Thread(target=thread_median,args=(tmpA,tmpB,i,num_threads,iterations,events))
        threads.append(t)
        t.start()

    tmpA, tmpB = tmpB, tmpA
    return tmpA



def thread_median(tmpA,tmpB,i,num_threads,iterations,e):
    filtering.median_3x3(tmpA,tmpB,i,num_threads)
    e[i][0].set()
    for j in range(1,iterations):
        if num_threads != 1:
            # if not np.all([event.isSet() for event in e[:][j-1]]):
            #     print "waiting on %d" % i
            #     e[i][j].wait()
            while(not e[i][j].isSet()): # while the event is not set,
                if i == 0:
                    if e[i+1][j-1].isSet() and e[i][j-1].isSet():
                        e[i][j].set()
                elif i == num_threads-1:
                    if e[i-1][j-1].isSet() and e[i][j-1].isSet():
                        e[i][j].set()
                else:
                    if e[i-1][j-1].isSet() and e[i+1][j-1].isSet() and e[i][j-1].isSet():
                        e[i][j].set()

        filtering.median_3x3(tmpA,tmpB,i,num_threads)
        tmpA[i::num_threads,:],tmpA[i::num_threads,:]=tmpA[i::num_threads,:],tmpA[i::num_threads,:]



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
    print from_cython-from_numpy
    assert np.all(from_cython == from_numpy)

    with Timer() as t:
        new_image = py_median_3x3(input_image, 10, 8)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()

# def py_median_3x3(image, iterations=10, num_threads=1):
#     ''' repeatedly filter with a 3x3 median '''
#     tmpA = image.copy()
#     tmpB = np.empty_like(tmpA)
#
#     for i in range(iterations):
#         filtering.median_3x3(tmpA, tmpB, 0, 1)
#         # swap direction of filtering
#         tmpA, tmpB = tmpB, tmpA
#
#     return tmpA
