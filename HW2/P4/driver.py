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

def py_median_3x3_ori(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    for i in range(iterations):
        filtering.median_3x3(tmpA, tmpB, 0, 1)
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA

    return tmpA

def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    # Set buffers for input and output images
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    # Initilize Events [num_threads x iterations]
    # it will be used to control cooperation between threads
    events = []
    for idx in range(num_threads*iterations):
        events = events + [threading.Event()]
    events = np.reshape(events, [num_threads,iterations])
            
    # Multithreading
    threads = []
    for threadidx in range(num_threads):
        th = threading.Thread(target=parallel_worker,\
                              args=(tmpA, tmpB, iterations, num_threads, events, threadidx))
        threads = threads + [th]
        th.daemon = True  # exit even when this thread is alive
        th.start()
        
    # Wait for all threads to be done
    for thread in threads:
        thread.join()

    return tmpA

def parallel_worker(tmpA, tmpB, iterations, num_threads, events, threadidx):
    for iter in range(iterations):
        # Need to check if n-1,n,n+1 threads are done at iter-1
        # before n thread is starting to work at iter
        # 
        if iter == 0:
            pass
        else:
            if threadidx == 0:
                events[threadidx,iter-1].wait()
                # except when num_threads = 1
                if num_threads != 1:
                    events[threadidx+1,iter-1].wait()
            elif threadidx == num_threads-1:
                events[threadidx-1,iter-1].wait()
                events[threadidx,iter-1].wait()
            else:
                events[threadidx-1,iter-1].wait()
                events[threadidx,iter-1].wait()
                events[threadidx+1,iter-1].wait()
    
        ### median_3x3(input_image, output_image, offset, step):
        # "offset" should be set for corresponding thread
        # "step" should be set based on number of threads
        filtering.median_3x3(tmpA, tmpB, threadidx, num_threads)
        # When the main work is done for [threadidx,iter], set flag becomes true
        # So that next jobs that is dependent on it can start
        events[threadidx,iter].set()
    
        # Swap for the next work
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

    
    # Thread 1 using original version
    with Timer() as t:
        new_image = py_median_3x3_ori(input_image, 10, 1)
    print("[base] Thread 1 : {} seconds for 10 filter passes.".format(t.interval))
    
    # Thread 1
    with Timer() as t:
        new_image = py_median_3x3(input_image, 10, 1)
    print("Thread 1 : {} seconds for 10 filter passes.".format(t.interval))

    # Thread 2
    with Timer() as t:
        new_image = py_median_3x3(input_image, 10, 2)
    print("Thread 2 : {} seconds for 10 filter passes.".format(t.interval))
    
    # Thread 4
    with Timer() as t:
        new_image = py_median_3x3(input_image, 10, 2)
    print("Thread 4 : {} seconds for 10 filter passes.".format(t.interval))    

    # Thread 8
    with Timer() as t:
        new_image = py_median_3x3(input_image, 10, 8)
    print("Thread 8 : {} seconds for 10 filter passes.".format(t.interval))
    
    # Thread 16
    with Timer() as t:
        new_image = py_median_3x3(input_image, 10, 8)
    print("Thread 16 : {} seconds for 10 filter passes.".format(t.interval))
    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')
    
    pylab.show()
