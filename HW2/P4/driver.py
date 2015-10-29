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
from Queue import Queue

def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)
    Threads = []
    Events = [] #np.empty([num_threads,iterations])
    
    #Create 2D array of events.
    for i in range(iterations):
        for n in range(num_threads):
            Events += [threading.Event()]
    Events = np.array(Events).reshape((num_threads,iterations))
    
    #Create separate threads and set them loose on the image!
    for n in range(num_threads):
        thread = threading.Thread(target=parallizeImage, args=(n, num_threads, tmpA, tmpB, Events, iterations))
        thread.start()
        Threads += [thread]
    for t in Threads:
        t.join()
        
    #Return filtered image
    return tmpA
    
def parallizeImage(threadNum, num_threads, tmpA, tmpB, Events, iterations=10):
    for i in range(iterations):
        #Check events, create checks for conditions. i==0, n==num_threads, etc...
        if num_threads >1:
            if i == 0:
                pass
            elif threadNum == 0:
                Events[threadNum+1,i-1].wait()
            elif threadNum == num_threads-1:
                Events[threadNum-1, i-1].wait()
            else:
                Events[threadNum-1,i-1].wait()
                Events[threadNum+1,i-1].wait()
            
        filtering.median_3x3(tmpA, tmpB, threadNum, num_threads)
        #Change event flag
        Events[threadNum,i].set()
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
