
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
    
    #Initialize one thread for each num_threads, and
    #a conditional variable for each thread/iteration combination.
    #This will be useful when we have to do our checks for n-1 and n+1
    #threads in interation i-1 later. 

    allThreads = []
    conditions = [[None for i in xrange(iterations)] for j in xrange(num_threads)]

    for n in xrange(num_threads):
        for i in xrange(iterations):
            cond = threading.Condition()
            conditions[n][i] = cond

    #For each value in num_threads, create a thread and check if it is the first thread
    #or last thread or just a normal thread. This probably could be done without so many checks here/
    #in a less verbose mannerbut since this is only done at initialization I don't think it leads to drastic 
    #performance differences so I just kept it anyway. 

    for n in xrange(num_threads):
        if n == 0:
            firstThread = True
            lastThread = False
            thread = threading.Thread(target=parallel_helper, args=(iterations, num_threads, n, firstThread, lastThread, conditions, tmpA, tmpB))
            allThreads.append(thread)
            thread.start()
        elif n == num_threads-1:
            lastThread = True
            firstThread = False
            thread = threading.Thread(target=parallel_helper, args=(iterations, num_threads, n, firstThread, lastThread, conditions, tmpA, tmpB))
            allThreads.append(thread)
            thread.start()
        else:
            lastThread = False
            firstThread = False
            thread = threading.Thread(target=parallel_helper, args=(iterations, num_threads, n, firstThread, lastThread, conditions, tmpA, tmpB))
            allThreads.append(thread)
            thread.start()

    #Join the threads once that is all done. 
    for thrd in allThreads:
        thrd.join()

    return tmpA

def parallel_helper(iterations, num_threads, threadNumber, firstThread, lastThread, conditions, tmpA, tmpB):
    for i in range(iterations):
        #When you start the iteration, make sure you grab a lock for it
        #from the relevant conditional variable.
        lock = conditions[threadNumber][i]
        lock.acquire()

        #Depending on if you are the First Thread or the last thread, check the previous
        #thread or the next thread or both in the previous iteration and make sure they are done.
        #The reason for the try/except is explained in P4.txt, but the basic reasoning is that
        #if the thread is still locked, then it will continue to wait, and if it isn't still locked
        #then it will throw an error when you try to "wait" on it, so we just pass and continue
        #on our day.
        
        if firstThread:
            try:
                conditions[0][i-1].wait()
            except:
                pass
        elif lastThread:
            try:
                conditions[num_threads-1][i-1].wait()
            except:
                pass
        else:
            try:
                conditions[threadNumber+1][i-1].wait()
                conditions[threadNumber-1][i-1].wait()
            except:
                pass
        filtering.median_3x3(tmpA, tmpB, threadNumber, num_threads)
        lock.notify_all()
        lock.release()
        tmpA, tmpB = tmpB, tmpA
        #return tmpA



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

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
