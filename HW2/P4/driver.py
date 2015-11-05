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

#Original:
"""
def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    for i in range(iterations):
        filtering.median_3x3(tmpA, tmpB, 0, 1)
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA

    return tmpA
"""

def worker(tmpA, tmpB, events, iterations, thread_id, num_threads):
    
    for i in range(iterations):
        
        filtering.median_3x3(tmpA, tmpB, thread_id, num_threads)
        
        events[i, thread_id].set()

        if num_threads==1: 
            pass #events[i].clear()
        elif num_threads==2: 
            if thread_id == 0: events[i,1].wait()
            else: events[i,0].wait() #
            # Don't seem to need this.  Why not?
	else:
            if thread_id==0: 
                events[i,1].wait()
                events[i,num_threads-1].wait()
            elif thread_id==num_threads-1:
                events[i,0].wait()
                events[i,thread_id-1].wait()
            else:
                events[i,thread_id-1].wait()
                events[i,thread_id+1].wait()
       
        tmpA, tmpB = tmpB, tmpA
    
def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)
    threads = []
    events = []
    
    # Create an array of events for each thread and iteration combo
    for iteration in range(iterations*num_threads):
        events.append(threading.Event())
    events = np.array(events).reshape([iterations,num_threads])

    for thread_id in range(num_threads):
        t = threading.Thread(target=worker, args=(tmpA, tmpB, events, iterations, thread_id,  num_threads))
        threads.append(t)
        t.start()
    
    for t in range(num_threads):
        threads[t].join()

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
        new_image = py_median_3x3(input_image, 10, 8)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
