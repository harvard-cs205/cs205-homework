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

def median_new(tmpA, tmpB, th_idx, num_threads,iterations):

    global Events_mat
    
    for i in range(iterations):
        #All threads can start at the same time in iteration 1 as they are not depending on any other.
        if i == 0:
            filtering.median_3x3(tmpA, tmpB, th_idx, num_threads)
            tmpA, tmpB = tmpB, tmpA
            
            #Setting that iteration 1 was completed for that particular thread
            Events_mat[th_idx,0].set()
            
        else:
            # We note that thread 1 (idx 0) only needs to check if thread 2 is done at the previous iteration to start
            if th_idx == 0:
            #We need to account for the case where have only 1 thread (serial) working, so we only wait for the completion of the previous iteration if we have more than 1 thread
                if num_threads > 1:
                    Events_mat[1,i-1].wait()
                filtering.median_3x3(tmpA, tmpB, th_idx, num_threads)
                
            # We note that thread n (idx n-1) only needs to check if thread n-1 is done at the previous iteration to start
            elif th_idx == (num_threads - 1):
        
                Events_mat[th_idx-1,i-1].wait()
                filtering.median_3x3(tmpA, tmpB, th_idx, num_threads)
            
            else:
        
                Events_mat[th_idx-1,i-1].wait()
                Events_mat[th_idx+1,i-1].wait()
                filtering.median_3x3(tmpA, tmpB, th_idx, num_threads)
            
            #Swapping to update:
            tmpA, tmpB = tmpB, tmpA
            #Setting that iteration i is completed for that particular thread
            Events_mat[th_idx,i].set()
    
  
            
def py_median_3x3(image, num_threads, iterations=10):
    ''' repeatedly filter with a 3x3 median '''
    global Events_mat 
    
    #Define a matrix of events of shape: nrow = num_threads, ncol = iterations
    Events_mat = np.array([[threading.Event()for x in range(iterations)] for x in range(num_threads)] )
    
    #Still not sure to understand the point Thouis made in @388 on piazza. My code is working, but I'm not sure why it is the case. It started with a very naive approach
    # but worked by creating tmpA and tmpB here. But thinking more about it, if my target median function updates for each thread, is the reason why we don't observe
    # corruption is that the median_3x3 makes sure each thread changes particular lines in the image? This is the approach I used but would love for someone to correct
    # me if I didn't get it !
     
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)
    
    #Creating a list to store the threads
    threads = range(num_threads)
    
    #Initialising the threads
    for th_idx in range(num_threads):
       
        threads[th_idx] = threading.Thread(target=median_new,args=(tmpA, tmpB, th_idx, num_threads,iterations))
        threads[th_idx].start()
    
    #Closing the threads
    for thread in threads:
        thread.join()
        
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

    #pylab.imshow(input_image)
    #pylab.title('original image')

    #pylab.figure()
    #pylab.imshow(input_image[1200:1800, 3000:3500])
    #pylab.title('before - zoom')

    # verify correctness
    from_cython = py_median_3x3(input_image, 5, 2)
    from_numpy = numpy_median(input_image, 2)
    assert np.all(from_cython == from_numpy)
    
    
   
    #with Timer() as t:
    #    new_image = py_median_3x3(input_image, 4, 10)

    #pylab.figure()
    #pylab.imshow(new_image[1200:1800, 3000:3500])
    #pylab.title('after - zoom')

    #print("{} seconds for 10 filter passes.".format(t.interval))
    #pylab.show()
    
    diffnum = [1,2,4,8]
    results = []
    for num in diffnum:
        with Timer() as t:
            new_image = py_median_3x3(input_image, num, 10)
        results.append(t.interval)
    
    plt.plot(diffnum,results,'ro')
    plt.xlabel('Number of threads')
    plt.ylabel('Time (s)')
    plt.title('Performance of py_median_3x3 for different number of threads')
    plt.xlim(0.5,8.5)
    plt.ylim(0.5,3.5)
    plt.show()