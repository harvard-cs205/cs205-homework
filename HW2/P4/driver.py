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

def filter_func(tmpA, tmpB, offset, step, worker_event, main_event, iterations):
    for i in range(iterations):
      filtering.median_3x3(tmpA, tmpB, offset, step)
      worker_event.set()
      main_event.wait()
      tmpA, tmpB=tmpB, tmpA

def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)
    worker_eventList=[]

    # worker threads events
    for i in range(num_threads): 
       worker_eventList.append(threading.Event())

    # main thread event, for waking up worker threads
    main_event=threading.Event()

    # start worker threads
    for threadidx in range(num_threads):
           th=threading.Thread(target=filter_func,args=(tmpA,tmpB,threadidx,num_threads,worker_eventList[threadidx],main_event,iterations))
           th.start()    

    # monitoring worker threads to acheive synchronization  
    for i in range(iterations):
        for e in worker_eventList:
           e.wait()
        for e in worker_eventList:
           e.clear() 
        # wake all worker threads
        main_event.set()
        # reset main_event
        main_event.clear()
           
    return tmpA

def numpy_median(image, iterations=10):
    ''' filter using numpy '''
    for i in range(iterations):
        padded = np.lib.pad(image, 1, mode='edge')
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
    #from_cython = py_median_3x3(input_image, 2, 5)
    #from_numpy = numpy_median(input_image, 2)
    #assert np.all(from_cython == from_numpy)

    with Timer() as t:
        new_image = py_median_3x3(input_image, 10, 2)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
