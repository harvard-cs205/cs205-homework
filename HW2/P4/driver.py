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
import threading as th

def per_thread_median(A,B,this_thr,evs,I):
	''' Runs Cython median_3x3() with thread-private offset
			
		Each thread waits until the previous iterations of threads controlling
		thread_num-1, thread_num, and thread_num+1 have finished. 
		
		"Finished" is signaled by the Event variable in evs[] for that particular thread on 
		the previous iteration (i-1).
		
		Example: 
			For current iteration i, with thread_index n, where 0 > n > num_threads-1:
			We wait to begin median_3x3 until:
				evs[ n-1, i-1 ] is set to True
				evs[ n  , i-1 ] is set to True
				evs[ n+1, i-1 ] is set to True
	'''
	num_threads = step = evs.shape[0]
	
	for i in range(I):
		
		'''If this_thr is index 0 or max-thread, wrap-around prev/next thread index
			Eg. this_thr == 0 --> prev_thread = max-thread idx
		
			NOTE: If only 2 threads, this should still be fine, 
				  it's just that two out of the three checks will be for the same Event. 
				  
				  Eg. For 2 threads, for thread 0, next_thr = 1, prev_thr = 1, 
				  so both will wait for same Event to become true.
		'''    
		prev_thr = this_thr-1 if this_thr-1 >= 0 else num_threads-1
		next_thr = this_thr+1 if this_thr+1 < num_threads else 0
		
		# for i==0 no checking is necessary
		if i: 
			# wait until all nearby threads have finished iteration i-1
			evs[ prev_thr, i-1 ].wait()  
			evs[ this_thr, i-1 ].wait() 
			evs[ next_thr, i-1 ].wait() 

		filtering.median_3x3(A, B, this_thr, step)
		A, B = B, A
		
		# set Event == True for current thread, current iteration 
		evs[this_thr,i].set()
		
def py_median_3x3(image, iterations=10, num_threads=1):
	''' repeatedly filter with a 3x3 median '''
	tmpA = image.copy()
	tmpB = np.empty_like(tmpA)

	# initialize Events array, shape: [num_threads x iterations]
	events = np.empty( (num_threads, iterations), dtype=th._Event )
	
	# populate Events array
	for i in xrange(num_threads):
		events[i,:] = [th.Event() for _ in xrange(iterations)]   
	
	# initialize threads array, shape: [num_threads]
	threads = np.empty(num_threads,dtype=th.Thread)

	# populate threads array
	for start in range(num_threads):
		threads[start] = th.Thread(name='t{}'.format(start), 
								   target=per_thread_median, 
								   args=(tmpA, tmpB, start, events, iterations)
								  )
		threads[start].start()
	
	# reunite threads into master, use .join() to make sure all threads have completed
	for ix in range(num_threads):
		threads[ix].join()
		
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

	iter_ct = 10
	for n_thread in [1,2,4]:

		from_cython = py_median_3x3(input_image, iter_ct, n_thread)
		from_numpy = numpy_median(input_image, iter_ct)
		assert np.all(from_cython == from_numpy)

		with Timer() as t:
			new_image = py_median_3x3(input_image, iter_ct, n_thread)

		print("{} THREADS: {} seconds for 10 filter passes.".format(n_thread,t.interval))
