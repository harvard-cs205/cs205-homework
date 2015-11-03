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

logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s] (%(threadName)-10s) %(message)s',
                    )

# ----------------------------------------------------------------

# FUNCTION GIVEN

# ----------------------------------------------------------------


def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy() 
    tmpB = np.empty_like(tmpA)

    for i in range(iterations):
        filtering.median_3x3(tmpA, tmpB, 0, 1)
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA

    return tmpA
# ----------------------------------------------------------------


# ----------------------------------------------------------------

# FUNCTIONS FOR DOING THE WORK BY CREATING NEW THREADS AT EACH ITERATIONS AND
# JOINING THEM ONCE THE ITERATION IS OVER AND THEN SWAP THE MATRICES (NOT THE
# EXTRA CREDIT QUESTION)

def worker(tmpA, tmpB, iterations, thread_id,num_threads ):

    
    filtering.median_3x3(tmpA,tmpB,thread_id,num_threads)           

    return tmpA, tmpB


def py_median_3x3_syn(image,iterations=10, num_threads=1):
    
    tmpA = image.copy() 
    tmpB = np.empty_like(tmpA)


    for i in range(iterations):

        # Create the treads, append them into a list
        # this way, it will be easier to join them at the end
        threads = []
        for thread_id in range(num_threads):
            # create the threads, pass the function and the inputs
            t = threading.Thread(target=worker, 
            args=(tmpA, tmpB, iterations, thread_id, num_threads))
            threads.append(t)
        # all threads start working now
        print 'threads start working now'
        map(lambda t:t.start(), threads)
        # join all the threads at each iteration
        print 'now joining all the threads'
        map(lambda t: t.join(),threads)
        # then swap tmpA and tmpB
        tmpA, tmpB = tmpB, tmpA 

    return tmpA

# ----------------------------------------------------------------


# ----------------------------------------------------------------

# ATTEMPT TO DO THE EXTRA CREDIT QUESTION BY DEFINING
# A LARGE MATRIX WITH THREADS AS ROWS AND ITERATIONS AS COLUMNS


def py_median_3x3_threads(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''

    tmpA = image.copy() 
    tmpB = np.empty_like(tmpA)

    events_matrix = [threading.Event() for _ in range(num_threads*iterations)]
    # set every events to clear (isSet() == False)
    map(lambda t: t.clear(), events_matrix)
    events_matrix = np.reshape(events_matrix,(num_threads,iterations))
    events_sum = np.zeros((num_threads,iterations))


    # Create the treads, append them into a list
    # this way, it will be easier to join them at the end
    threads = []
    for thread_id in range(num_threads):
    	# create the threads, pass the function and the inputs

      	t = threading.Thread(target=filter_image, 
            args=(tmpA, tmpB, iterations, thread_id, num_threads,
             events_matrix, events_sum))
        threads.append(t)
    # all threads start working now
    print 'threads start working now'
    map(lambda t:t.start(), threads)
    # kill all the threads because the work is done
    print 'now joining all the threads'
    map(lambda t: t.join(),threads)

    return tmpA
    
def filter_image(tmpA, tmpB, iterations, thread_id,num_threads, events_matrix,events_sum ):

    
	for i in range(iterations):
		# look when iteration >0
		if i>0:

			while not (events_matrix[thread_id, i-1].isSet() 
				and events_matrix[max(thread_id-1,0),i-1].isSet()
				and events_matrix[min(thread_id +1, num_threads-1),i-1].isSet()):
	 	 		 print('thread_id {}, iteration {} is waiting'.format(thread_id,i))
	 	 		 # wait a bit for the previous iteration to be fully completed
	 	 		 events_matrix[thread_id,i].wait(1)
	 	 		 print('thread_id {}, iteration {} is no longer waiting'.format(thread_id,i))



		filtering.median_3x3(tmpA,tmpB,thread_id,num_threads)  			


		# swap direction of filtering once iteration is done
		events_matrix[thread_id,i].set()
		print('thread_id {}, iteration {} is set {}'.format(thread_id,i,events_matrix[thread_id,i].isSet()))
		tmpA, tmpB = tmpB, tmpA


	return tmpA

# ----------------------------------------------------------------

# ----------------------------------------------------------------

# THE FUNCTION GIVEN THAT RUNS THE CODE IN PYTHON
# ----------------------------------------------------------------


def numpy_median(image, iterations=10):
    ''' filter using numpy '''
    for i in range(iterations):
        padded = np.pad(image, 1, mode='edge')
        stacked = np.dstack((padded[:-2,  :-2], padded[:-2,  1:-1], padded[:-2,  2:],
                             padded[1:-1, :-2], padded[1:-1, 1:-1], padded[1:-1, 2:],
                             padded[2:,   :-2], padded[2:,   1:-1], padded[2:,   2:]))
        image = np.median(stacked, axis=2)

    return image
# ----------------------------------------------------------------


if __name__ == '__main__':
    input_image = np.load('image.npz')['image'].astype(np.float32)

    pylab.gray()

    pylab.imshow(input_image)
    pylab.title('original image')

    pylab.figure()
    pylab.imshow(input_image[1200:1800, 3000:3500])
    pylab.title('before - zoom')

    # verify correctness
    # I am using the sync version here
    from_cython = py_median_3x3_syn(input_image, iterations=2, num_threads=3)
    from_numpy = numpy_median(input_image, 2)
    assert np.all(from_cython == from_numpy)
    print 'FIRST TEST IS PASSED'
    print 'NOW DOING THE REAL IMAGE PROCESSING'


    with Timer() as t:
        # I am using the sync version here
        new_image = py_median_3x3_syn(input_image, 10, 4)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
