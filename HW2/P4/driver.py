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
    locks = [None]*num_threads
    calc_count = [0]*num_threads
    swap_count = [0]*num_threads
    threads = [None]*num_threads
    for i in range(num_threads):
        locks[i] = threading.Condition()
    
    for i in range(num_threads):
        threads[i] = threading.Thread(target=median_worker, args=(i, iterations, tmpA, tmpB, locks, calc_count, swap_count))
        threads[i].start()
        
    for t in threads:
        t.join()
        
    return tmpA

def median_worker(thread_num, iterations, tmpA, tmpB, locks, calc_count, swap_count):
    neighbor_threads = set([thread_num-1 if thread_num > 0 else len(locks)-1, thread_num + 1 if thread_num < len(locks) - 1 else 0])
    for i in range(iterations):  
        #can't calculate until neighbor threads have been copied to the right iteration
        #if they haven't been copied yet, wait on corresponding locks until notified
        for j in neighbor_threads:
            if j != thread_num:
                locks[j].acquire()
                while True:
                    if swap_count[j] == swap_count[thread_num]:
                        break
                    locks[j].wait()  
        for j in neighbor_threads:
            if j != thread_num:
                locks[j].release()       
        with locks[thread_num]:
            filtering.median_3x3(tmpA, tmpB, thread_num, len(locks))
            calc_count[thread_num] += 1
            locks[thread_num].notifyAll()
            #print "{}: Calculated {}".format(thread_num, i)
        
        #can't swap until neighbor threads have calculated to the right iteration
        #if they haven't been calculated yet, wait on the corresponding locks until notified
        for j in neighbor_threads:
            if j != thread_num:
                locks[j].acquire()
                while True:
                    if calc_count[j] == calc_count[thread_num]:
                        break
                    locks[j].wait()
        for j in neighbor_threads:
            if j != thread_num:
                locks[j].release() 
        with locks[thread_num]:
            for j in range(thread_num, len(tmpA), len(locks)):
                tmpA[j],tmpB[j] = tmpB[j],tmpA[j]
            swap_count[thread_num] +=1
            locks[thread_num].notifyAll()
            #print "{}: Swapped {}".format(thread_num, i)

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
    from_cython = py_median_3x3(input_image,10, 20)
    from_numpy = numpy_median(input_image, 10)
    assert np.all(from_cython == from_numpy)

    thread_arr = [1,2,4,8]
    thread_times = []
    for threads in thread_arr:
        with Timer() as t:
            new_image = py_median_3x3(input_image, 10, threads)    
        print("{} seconds for 10 filter passes.".format(t.interval))
        thread_times.append(t.interval)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')
    pylab.show()
    
    pylab.figure()
    pylab.plot(thread_arr, thread_times)
    pylab.xlabel("Threads")
    pylab.ylabel("Seconds")
    pylab.title('Thread Performance')
    pylab.show()
