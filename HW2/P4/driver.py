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

# original code
def t_py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    for i in range(iterations):
        filtering.median_3x3(tmpA, tmpB, 0, 1)
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA

    return tmpA

# my code with parallelize thread

def process_lines(tmpA, tmpB, no_thread,num_threads,locks):
    # process  no. line 
    line = no_thread
    while  line  < tmpA.shape[0] :       
        # line,line+1,line-1 lock is set. No processing 0th line and last line
	if line == 0:
	    locks[line].acquire()
	    locks[line+1].acquire()
	elif line == tmpA.shape[0]-1:
	    locks[line-1].acquire()
	    locks[line].acquire()		    
        else:
	    locks[line-1].acquire()	
	    locks[line].acquire()
	    locks[line+1].acquire()	
        # process no. line	
        filtering.median_3x3(tmpA, tmpB, line, tmpA.shape[0])
        # line,line+1,line-1 lock is released. No processing 0th line and last line
        if line == 0:
	    locks[line].release()
	    locks[line+1].release()	
	elif line == tmpA.shape[0]-1:
	    locks[line-1].release()	
	    locks[line].release()			
	else:
	    locks[line-1].release()	
	    locks[line].release()
	    locks[line+1].release()
        line = line + num_threads			

def py_median_3x3(image, iterations=10, num_threads=4):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)
    locks = []
    #threads = []
    # locks are ready for the number of line in the 2D image array, 
    # that is one line to one Lock
    for l in range(image.shape[0]):
	locks += [threading.Lock()]
    #print image.shape[0],len(locks)
		
    for i in range(iterations):
	#generating num_thread threads
        threads = []
        for t in range(num_threads):
	    threads += [threading.Thread(target = process_lines, args = (tmpA,tmpB,t,num_threads,locks))]
        for t in range(num_threads):
	    threads[t].start()	
	#collecting num_thread threads
        for t in range(num_threads):
	    threads[t].join()			
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA
    return tmpA
# my code with parallelized thread ends.		

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
