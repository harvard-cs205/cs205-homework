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

#we have N threads, and each thread operates on a bunch of rows of the image. each thread should run the filter function on its rows. The arguments should be 
#filtering.median_3x3(tmpA, tmpB, threadid, num_threads)
# we need to make sure that thread n waits to start iteration i until thread n-1, n, and n+1 complete iteration i-1. 
#using events -- use an event per thread per iteration, so num_threads*iterations events. Use event[thread i][iter j-1].wait()
#           event[thread i-1][iter j-1].wait()
#           event[thread i+1][iter j-1].wait()
#           do filter
#           event[thread i][iter j].set()

    #create events
    events=[[threading.Event() for i in range(iterations)] for j in range(num_threads)]
    threads=[]
    for threadidx in range(num_threads):
        th=threading.Thread(target=apply_py_median,args=(tmpA, tmpB, iterations, threadidx,num_threads,events))
        threads.append(th)
        th.daemon = True # exit even when this thread is alive
        th.start()
    for threadidx in range(num_threads):
        threads[threadidx].join()
    return tmpA

def apply_py_median(tmpA, tmpB, iterations, threadidx,num_threads,events):

    for iter in range(iterations):
        #wait for neighboring events to finish
        if iter > 0: #don't wait for first iteration
            if threadidx > 0: #lower thread edge
                events[threadidx - 1][iter - 1].wait() 
            events[threadidx][iter - 1].wait() #this is probably unnecessary
            if threadidx < num_threads - 1: #upper thread edge
                events[threadidx + 1][iter - 1].wait()
        #print('Thread {}, Iteration {} of {}'.format(threadidx,iter,iterations))
        #do filter
        tmpB=np.zeros_like(tmpA)
        print('Before filter: Iter {}, tmpA[10,10]={}, tmpB[10,10]={}'.format(iter,tmpA[300,300],tmpB[300,300]))            
        filtering.median_3x3(tmpA, tmpB, threadidx, num_threads)
        print('After filter: Iter {}, tmpA[10,10]={}, tmpB[10,10]={}'.format(iter,tmpA[300,300],tmpB[300,300]))            
        # swap direction of filtering, but make sure we only do it for the elements that were just filtered from this thread
        # for i in range(threadidx,tmpA.shape[0],num_threads):
        #     tmpA[i,:], tmpB[i,:] = tmpB[i,:], tmpA[i,:]
        #     if i==10:
                # print('Iter {}, tmpA[10,10]={}'.format(iter,tmpA[10,10]))
        tmpA = tmpB
        
        #print('completed')
        #mark current event as done
        events[threadidx][iter].set()

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
    # from_cython = py_median_3x3(input_image, 2, 1)
    # from_numpy = numpy_median(input_image, 2)
    # assert np.all(from_cython == from_numpy)
    #threads_to_try=[1,2,4,8]
    #for num_threads in threads_to_try:
    num_threads=1
    for passes in [10]:
        with Timer() as t:
            new_image = py_median_3x3(input_image, passes, num_threads)

        pylab.figure()
        pylab.imshow(new_image[1200:1800, 3000:3500])
        pylab.title('{} passes - zoom'.format(passes))

        print("{} seconds for {} filter passes and {} threads.".format(t.interval,passes,num_threads))
    pylab.show()