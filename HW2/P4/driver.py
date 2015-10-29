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
import matplotlib.pyplot as plt

def threaded_work(iterations,num_threads,n,tmpA,tmpB,Events):
    init = 0
    filtering.median_3x3(tmpA, tmpB, n, num_threads)
    tmpA, tmpB = tmpB, tmpA
    Events[n,init].set()

    if num_threads > 1:
        for i in range(1,iterations):

            if n == 0:
                Events[n+1,i-1].wait()
            elif n == num_threads-1:
                Events[n-1,i-1].wait()
            else:
                Events[n+1,i-1].wait()
                Events[n-1,i-1].wait()

            filtering.median_3x3(tmpA, tmpB, n, num_threads)
            # swap direction of filtering
            tmpA, tmpB = tmpB, tmpA
            Events[n,i].set()

        
        
def py_median_3x3(image, iterations=10, num_threads=1):

    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)
    thread_list=[] 
    Events = []
    
    for n in range(num_threads*iterations):
        Events.append(threading.Event())
    Events = np.array(Events).reshape(num_threads,iterations)
     
    for thrd in range(num_threads):
        t = threading.Thread(target=threaded_work,args=(iterations,num_threads,thrd,tmpA,tmpB,Events))
        thread_list.append(t)
        t.start()
    for thrd in thread_list:
        thrd.join()
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
    
    #times=[]
    #thread_count = [1,2,4]
    #with Timer() as t:
        #for tc in thread_count:
            #new_image = py_median_3x3(input_image, 10, tc)
            #times.append(t.interval)
            

    with Timer() as t:
            new_image = py_median_3x3(input_image, 10, 4)
            
    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()    
    
    #plt.bar(options,times)
    
    