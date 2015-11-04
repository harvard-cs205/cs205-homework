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


def py_median_3x3_0(image, iterations):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    for i in range(iterations):
        filtering.median_3x3(tmpA, tmpB, 0, 1)
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA

    return tmpA

def applyFilter(input_image,output_image,offset,step,controlS):
    filtering.median_3x3(input_image,output_image,offset,step)
    controlS.set()
    

def py_median_3x3(image, iterations, num_threads):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)
    control = [threading.Event() for ii in xrange(0,num_threads)]
    for i in xrange(iterations): 
        for threadidx in range(num_threads):
            th = threading.Thread(target=applyFilter,
                                  args=(tmpA,tmpB,threadidx,num_threads,control[threadidx]))
            th.start()
        for ii in xrange(0,num_threads):
            control[ii].wait()
        tmpA, tmpB = tmpB, tmpA
        for ii in xrange(0,num_threads):
            control[ii].clear()
        
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
    input_image = np.copy(input_image)
    
    pylab.figure()
    pylab.imshow(input_image[1200:1800, 3000:3500])
    #pylab.imshow(input_image)
    pylab.title('before - zoom')
    pylab.show()
    
    # verify correctness
    #from_cython = py_median_3x3(input_image, 2, 5)
    #from_numpy = numpy_median(input_image, 2)

    num_iterations = 10
    num_threads = 1

    print "Num Threads: ", num_threads    

    from_cython0 = py_median_3x3_0(input_image, num_iterations)
    from_cython = py_median_3x3(input_image, num_iterations, num_threads)
    from_numpy = numpy_median(input_image, num_iterations)
    #print from_cython0[0][0:10] 
    #print from_cython[0][0:10]
    #print [sum(np.array(from_cython[ii] == from_cython0[ii])) for ii in xrange(0,600)]
    #print sum(sum(np.array(from_cython[ii] == from_cython0[ii])) for ii in xrange(0,600))
    assert np.all(from_cython == from_cython0)
    assert np.all(from_cython == from_numpy)
  
    with Timer() as t:
        #new_image = py_median_3x3(input_image, 10, 8)
        new_image = py_median_3x3(input_image, num_iterations, num_threads)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    #pylab.imshow(new_image)
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
  
