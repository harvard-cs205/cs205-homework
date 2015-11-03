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

    events = []
    threads = []
    for iter_i in range(iterations):
        print(iter_i)

        events.append([])
        threads.append([])
        for thread_i in range(num_threads):            
            # Create events to synchronize threads
            events[iter_i].append(threading.Event())           
            
            t = threading.Thread(target=threaded_work,
                                 args=(tmpA, 
                                       tmpB, 
                                       thread_i, 
                                       num_threads, 
                                       iter_i,
                                       events))
            t.start()
            threads[iter_i].append(t)
            
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA

    # Make sure all the the threads are complete before exiting
    for iter_i in range(iterations):
        for thread_i in range(num_threads):
            threads[iter_i][thread_i].join()
    return tmpA
    
def threaded_work(tmpA, tmpB, thread_i, num_threads, iter_i, events):
    # If this is not the first iteration,
    # identify which threads of the previous iteration 
    # need to be done before moving on.
    if iter_i > 0:
        back_thread = np.mod(thread_i - 1, num_threads)
        front_thread = np.mod(thread_i + 1, num_threads)

        #print("Checking for iter{}, thread{}, checking t{} and t{}".format(iter_i, thread_i, back_thread, front_thread))
        events[iter_i - 1][thread_i].wait()
        events[iter_i - 1][back_thread].wait()
        events[iter_i - 1][front_thread].wait()
        #print("Completed check for iter{}, thread{}, checked t{} and t{}".format(iter_i, thread_i, back_thread, front_thread))        
        
    filtering.median_3x3(tmpA, tmpB, thread_i, num_threads)
    
    # Notify other threads of completion
    events[iter_i][thread_i].set() 

# Note that this function modifies the original image
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

    n_iterations = 10
    num_threads_list = [1, 2, 4]
    perf_times = []
    new_images = []

    for num_threads in num_threads_list:
        with Timer() as t:
            from_cython = py_median_3x3(input_image, n_iterations, num_threads)

        perf_times.append(t.interval)
        new_images.append(from_cython)
        print("{} threads took {} sec".format(num_threads, t.interval))
        
    with Timer() as t:
        from_numpy = numpy_median(input_image, n_iterations)  
    original_time = t.interval
    print("Original took {} sec".format(t.interval))

    # verify correctness
    for thread_i in range(len(num_threads_list)):
        assert(np.all(from_numpy == new_images[thread_i]))


    pylab.figure()
    for thread_i in range(len(num_threads_list)):
        pylab.subplot(1, 4, thread_i + 2)
        pylab.imshow(new_images[thread_i][1200:1800, 3000:3500])
        pylab.title("{} threads".format(num_threads_list[thread_i]))
        pylab.xticks([])
        pylab.yticks([])
        
    pylab.subplot(1, 4, 1)
    pylab.imshow(new_images[thread_i][1200:1800, 3000:3500])
    pylab.title('Serial version')
    pylab.xticks([])
    pylab.yticks([])
    pylab.savefig('P4_image_correctness.png')
#    pylab.show()
    
    pylab.figure()
    pylab.bar(range(4), [original_time] + perf_times, color = ['k', 'b', 'b', 'b'])
    pylab.xticks(0.5 + np.arange(4), ['Numpy', '1 thread', '2 threads', '4 threads'])
    pylab.title('Run times for {} iterations'.format(n_iterations))
    pylab.ylabel('Run time (sec)')
    pylab.savefig('P4_perf_times.png')
#    plt.show()
    
    

#    print("{} seconds for 10 filter passes.".format(t.interval))
#    pylab.show()
