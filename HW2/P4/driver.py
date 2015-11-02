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


    #the original method provided by professors
    # for i in range(iterations):
    #    filtering.median_3x3(tmpA, tmpB, 0, 1)
    #    # swap direction of filtering
    #    tmpA, tmpB = tmpB, tmpA
    #
    # #multiple-threads method
    # #first,create an event for each (threads,iteration)
    # #event will be used for protecting communications between threads
    Events = []
    for x in range(num_threads):
        for iter in range(iterations):
            Events.append(threading.Event())
    #reshape the events, and access it in the future
    Events = np.array(Events).reshape((num_threads,iterations))

    #at the first iteration
    #we shall create all threads in the form of
    Threads=[]
    for x_thread in range(num_threads):
        thread = threading.Thread(target=parallelComputeImage,args=(x_thread,num_threads,tmpA,tmpB,Events,iterations))
        thread.start();
        Threads.append(thread)

    #we actually need to kill the last thread
    Threads[num_threads-1].join()
    return tmpA


#this function is used for processing image in multiple threads
def parallelComputeImage(x_thread,num_threads,tmpA,tmpB,Events,iterations):
    #create a function for processing image in multiple threads
    #all threads will run parallel

    for i in range(iterations):
        #before running threads, make sure everything is ready to run
        #We don't need to differentiate if its single thread/multiple threads here
        #we need to check if n has completed its calculation

        #the first iteration no need to wait for anything.
        #the first thread has no x_thread - 1
        #the last thread has no x_thread + 1
        #print "iteration time=%d" %i
        if(i != 0 and num_threads != 1):
            Events[x_thread,i-1].wait()
            #skip if it is the first x_thread
            if(x_thread!=0):
                Events[x_thread - 1,i-1].wait()
            #skip if it is the last x_thread
            if(x_thread!=num_threads - 1):
                Events[x_thread + 1, i-1].wait()

        #run filtering function(tmpA,tmpB,offset,step)
        #step----step here is the numThread, since we are creating every Nth line
        #offset---- here is Thread Number also, cuz we are starting at this line
        #x here is the number assigned to every thread in the previous loop
        filtering.median_3x3(tmpA, tmpB, x_thread, num_threads)
        #here we send out the signal to make sure
        Events[x_thread,i].set()
        #swap in the single thread, and run everything again until we hit the max_iteration
        tmpA, tmpB = tmpB, tmpA



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
        new_image = py_median_3x3(input_image, 10, 8)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
