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


# ORIGINAL CODE
# def py_median_3x3(image, iterations=10, num_threads=1):
#     ''' repeatedly filter with a 3x3 median '''
#     tmpA = image.copy()
#     tmpB = np.empty_like(tmpA)
    

#     for i in range(iterations):
#         filtering.median_3x3(tmpA, tmpB, 0, 1)
#         # swap direction of filtering
#         tmpA, tmpB = tmpB, tmpA

#     return tmpA


#Function that ensure that threads do not start an interation of filtering before the data for that 
#iteration is ready
def parallel_function(tmpA, tmpB, t, num_threads, events_list, iterations):
    for i in range(iterations):
        #We clear the event t
        events_list[t].clear()
        
        #We filter the image (only the rows t, t+4, t+4+4, etc)
        filtering.median_3x3(tmpA, tmpB, t, num_threads)
        
        #One the filtering is done we put a flag
        events_list[t].set()
        
        #We swap direction of filtering
        tmpA, tmpB = tmpB, tmpA
        
        for e in events_list:
            #We make sure that we are not waiting for ourselves
            if e!=t:
                #This blocks the events to go forward until all the events of that iteration are done
                e.wait()

#Extra credit. Function that allows threads to begin an interation of filtering asynchronously
def parallel_function_asynchronously(tmpA, tmpB, t, num_threads, events_list, iterations):
    #print "STARTING"
    for i in range(iterations):
        
        #We clear the event t
        events_list[t].clear()
        
        #We filter the image (only the rows t, t+4, t+4+4, etc)
        filtering.median_3x3(tmpA, tmpB, t, num_threads)
        
        #One the filtering is done we put a flag
        events_list[t].set()
        
        #We swap direction of filtering
        tmpA, tmpB = tmpB, tmpA
        
        #If the number of threads is 2 then we always need to wait for the other thread. If the number of threads is one then there is no need to wait.
        if num_threads<=2:
            l = 0
            for e in events_list:
                #We make sure that we are not waiting for ourselves
                
                if e!=t:
                    #This blocks the events to go forward until all the events of that iteration are done
                    e.wait()
                l+=1

        else:
            #It there are 3 threads or more we need to wait for the threads that are calculating the filter for rows t-1 and t+1.
            
            #We pay attention to the special cases where t=num_threads
            if t == num_threads-1:
                events_list[t-1].wait()
                events_list[0].wait()
                
            #and where t=0 
            elif t==0:
                events_list[num_threads-1].wait()
                events_list[t+1].wait()
            #We only wait for the events that are doing the rows t-1 and t
            else:
                events_list[t-1].wait()
                events_list[t+1].wait()
       
                
            

def py_median_3x3(image, iterations=10, num_threads=2):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)
    
    #We create a list of events: one event for each thread.
    events_list = []
    for th in range(num_threads):
        e = threading.Event()
        events_list.append(e)

    threads_list = []
    for th in range(num_threads):
        
        #We create num_threads threads and we append them to a list of threads
        t = threading.Thread(target=parallel_function_asynchronously, args=(tmpA, tmpB, th, num_threads, events_list, iterations))
        t.start()
        threads_list.append(t)
        
    #We make sure that all the threads are done before returning the final output
    for thread in threads_list:
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
        new_image = py_median_3x3(input_image, 10, 1)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
