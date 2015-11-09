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

import matplotlib.pyplot as plt


logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)


#Reference code for median implementation
'''
def py_median_3x3(image, iterations=10, num_threads=1):
    #repeatedly filter with a 3x3 median
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    for i in range(iterations):
        filtering.median_3x3(tmpA, tmpB, 0, 1)
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA

    return tmpA
'''

def filter(thread_num, tmpA, tmpB, offset, step, iterations, event_list):
    #All logging comments have been commented.
    #logging.debug('Starting thread')
    for i in range(iterations):
        #logging.debug('iteration:'+str(i))
        if i==0 or step == 1:
            #Initial first iteration where we set the locks for the next steps
            filtering.median_3x3(tmpA, tmpB, offset, step)
            tmpA,tmpB = tmpB, tmpA
            event_list[i][thread_num].set()
        else:
            if thread_num == 0:
                event_list[i-1][thread_num+1].wait()
                #logging.debug("lock release")
            elif thread_num == step-1:
                event_list[i-1][thread_num-1].wait()
                #logging.debug("lock release")
            else:
                event_list[i-1][thread_num-1].wait()
                event_list[i-1][thread_num+1].wait()
                #logging.debug("lock release")
            filtering.median_3x3(tmpA, tmpB, offset, step)
            event_list[i][thread_num].set()
            #logging.debug("lock set")
            tmpA,tmpB = tmpB, tmpA
            #logging.debug("swap")
    '''
    for i in range(iterations):
        filtering.median_3x3(tmpA, tmpB, offset, step)
        tmpA,tmpB = tmpB, tmpA
    '''

    

def py_median_3x3(image, iterations=10, num_threads=1):
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)
    event_list = []
    thread_list = [None]*num_threads
    #Making a list of events
    for i in range(iterations):
        inter_list = []
        for j in range(num_threads):
            inter_list.append(threading.Event())
        event_list.append(inter_list)  
    
    for k in range(num_threads):
        #Creating threads
        thread_list[k] = threading.Thread(target=filter,\
                args=(k, tmpA, tmpB, k, num_threads, iterations, event_list))
        thread_list[k].start()
        
    for k in range(num_threads):
        #Joining all threads
        thread_list[k].join()
        #logging.debug('Joining thread')    
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
    '''
    #Plotting code
    iterations = [10,20,30]
    n_threads = [1,2,4]
    times = []
    colors = ['r','g','b']
    plt.figure()
    k = 0
    for iter in iterations:
        times = []
        for n in n_threads:
            with Timer() as t:
                new_image = py_median_3x3(input_image, iter, n)
            times.append(t.interval)
        plt.scatter(n_threads,times,c=colors[k],label="Iterations:"+str(iter))
        k += 1
    plt.legend()
    plt.ylabel("Time taken(seconds")
    plt.xlabel("No. of threads")
    plt.title("Performance of Threads comparison")
    plt.savefig("Performance_eval")
    
    '''
    with Timer() as t:
        new_image = py_median_3x3(input_image, 10, 1)
    
    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()

