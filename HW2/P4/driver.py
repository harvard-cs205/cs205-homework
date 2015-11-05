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

# wait for n-1,n,n+1 to be done
def wait_to_start(e,n,i):
    if i==1:
        return

    # wait for n-1
    if n > 0:
        e[n-1].wait()

    # wait for n
    e[n].wait()

    # wait for n+1
    if n < len(e)-1:
        e[n+1].wait()

    # once all these are set, n can move on




def py_median_3x3(image, iterations=10, num_threads=4):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    """
    # without multithreading
    for i in range(iterations):
        filtering.median_3x3(tmpA, tmpB, 0, 1)
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA

    """

    # same thing now with threading

    # make array to hold events so you know when things are completed
    # an event for each corresponding thread
    events = [threading.Event() for n in range(num_threads)]

    # will go through several iterations
    for i in range(iterations):
        # each iteration has num_threads to update image
        for n in range(num_threads):

            # first iteration runs through everything
            # make threads only once
            if i == 0:
                # make thread
                t = threading.Thread(target=wait_to_start,
                                     args=(events,n,i))
                t.start()

                # filter
                filtering.median_3x3(tmpA, tmpB, n, num_threads)
                # set to signial that it is done
                events[n].set()



            #if n ==0:
                """
                if events[n].isSet() and events[n+1].isSet():
                elif events[n-1].isSet() and events[n].isSet() and events[n+1].isSet():
                """

            elif i>0:

                # clear n so it isn't set and others that depend
                # on it will wait
                events[n].clear()

                # filter and set to signal completion
                filtering.median_3x3(tmpA, tmpB, n, num_threads)
                events[n].set()


            # swap direction of filtering
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

    """
    # verify correctness
    from_cython = py_median_3x3(input_image, 2, 5)
    from_numpy = numpy_median(input_image, 2)
    assert np.all(from_cython == from_numpy)
    """

    with Timer() as t:
        new_image = py_median_3x3(input_image, 10, 8)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()

