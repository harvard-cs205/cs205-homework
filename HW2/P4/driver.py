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
    
    if num_threads == 1:
        for i in range(iterations):
            filtering.median_3x3(tmpA, tmpB, 0, num_threads)
            # swap direction of filtering
            tmpA, tmpB = tmpB, tmpA

    else:
        #Set up (num_threads,iterations) threading events with the internal flag as False initially
        times = num_threads*iterations 
        processes = np.array([threading.Event()]*times).reshape((num_threads,iterations))
       
        #Set up multiple threads
        threads = []  
        for thread_j in range(num_threads):
            x = threading.Thread(target=f, args=(thread_j, num_threads, tmpA, tmpB, processes, iterations))
            x.start()
            threads.append(x)
        for x in threads:
            x.join()
        
    #Return filtered image
    return tmpA
    
def f(j, num_threads, tmpA, tmpB, processes, iterations):
    for i in range(iterations):
        if i==0:
            pass
        else:
            # iteration i thread 0 after completed iteration i-1 thread 1
            if j == 0:
                processes[j+1,i-1].wait()
            # iteration i thread (num_threads-1) after completed iteration i-1 thread (num_thread-2)
            elif j == num_threads-1:
                processes[j-1,i-1].wait()
            # iteration i thread j after completed iteration i-1 thread j-1, j+1
            else:
                processes[j-1,i-1].wait()
                processes[j+1,i-1].wait()
           
        filtering.median_3x3(tmpA, tmpB, j, num_threads)
        #set the internal flag to true
        processes[j,i].set()
    	# swap direction of filtering
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

    with Timer() as t:
        new_image = py_median_3x3(input_image, 10, 8)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
