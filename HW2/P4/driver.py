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

    for i in range(iterations):
        filtering.median_3x3(tmpA, tmpB, 0, 1)
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA

    return tmpA



def py_median_3x3_threading(image, iterations, num_threads):
    
    ''' 
    
    repeatedly filter with a 3x3 median using threading
    
    '''
    
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)
    
    event = np.array([[threading.Event()]*iterations]*num_threads)
    
    for n in range(num_threads):
        thread = threading.Thread(target=worker, args=(tmpA, tmpB, iterations, n, 
                                                       num_threads, event))
        
        thread.start() #intiate threads
    thread.join() #join threads
    

    #make sure to pick out the appropriate thread for odd vs. even iteration
    
    if  iterations % 2 == 1:
        return tmpB
    else:
        return tmpA


def worker(in_coord, out_coord, iterations, threadid, thread_num, event_array):
    
'''
This function test ensures thread synchronization first evaluates to ensure all adjacent threads
have finished the previous iteration then executes the operation on the 
image.
'''

    for i in range(iterations):

        #pass if there is only one thread, no logic neccessary for serial
        if thread_num == 1: 
            pass
        #pass if this is only the first iteration, no thread sync neccssary
        elif i == 0:
            pass
        #if the thread is the first iteration only check the thread after, for previous iteration
        elif threadid == 0:
            event_array[threadid + 1,i - 1].wait()
        #if thread is the last, only check thread afterward, for previous iteration
        elif threadid == thread_num - 1:
            event_array[threadid - 1, i - 1].wait()
        #for middle threads check above and below threads, for previous iteration
        else:
            event_array[threadid - 1, i - 1].wait()#causes to current thread
            event_array[threadid + 1, i - 1].wait()# to wait on these threads 
                                                   #to be complete
                
        filtering.median_3x3(in_coord, out_coord, threadid, thread_num)
        event_array[threadid, i].set() #tell the event that this thread is complete
        in_coord, out_coord = out_coord, in_coord




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

    # pylab.gray()

    # pylab.imshow(input_image)
    # pylab.title('original image')

    # pylab.figure()
    # pylab.imshow(input_image[1200:1800, 3000:3500])
    # pylab.title('before - zoom')

    # verify correctness
    from_cython = py_median_3x3_threading(input_image, 2, 5)
    from_numpy = numpy_median(input_image, 2)
    assert np.all(from_cython == from_numpy), "Not equal to numpy"

    with Timer() as t_1:
        new_image_threading = py_median_3x3_threading(input_image, 10, 1)

    with Timer() as t_2:
        new_image_threading = py_median_3x3_threading(input_image, 10, 2)

    with Timer() as t_4:
        new_image_threading = py_median_3x3_threading(input_image, 10, 4)

    with Timer() as tt:
        new_image_no_threading = py_median_3x3(input_image, 10, 1)

    assert np.all(new_image_threading == new_image_no_threading), "not equal to non-threading version"

    # pylab.figure()
    # pylab.imshow(new_image_threading[1200:1800, 3000:3500])
    # pylab.title('after - zoom')

    print("One way parallelism: {} seconds for 10 filter passes with threading.".format(t_1.interval))
    print("Two way parallelism: {} seconds for 10 filter passes with threading.".format(t_2.interval))
    print("Four way parallelism: {} seconds for 10 filter passes with threading.".format(t_4.interval))
    print("Baseline: {} seconds for 10 filter passes without threading.".format(tt.interval))

    pylab.show()










