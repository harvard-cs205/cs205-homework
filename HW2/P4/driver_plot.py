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

import matplotlib.pyplot as plt


def py_median_3x3(image, iterations=10, num_threads=1):
    ''' repeatedly filter with a 3x3 median '''
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    threads = []

    # Create events that signals the threads to avoid collisions
    events = np.array([threading.Event()]*(iterations*num_threads)).reshape([num_threads, iterations])

    if num_threads==1:
        for i in range(iterations):
            filtering.median_3x3(tmpA, tmpB, 0, 1)
            # swap direction of filtering
            tmpA, tmpB = tmpB, tmpA
    else:
        for tid in range(num_threads):
            t = threading.Thread(target=filterHelper, 
                                 args=(iterations, tid, num_threads, tmpA, tmpB, events))
            threads.append(t)
            t.start()
        for tid in range(num_threads):
            threads[tid].join()

    return tmpA

def filterHelper(iterations, tid, num_threads, tmpA, tmpB, events):

    # The first iteration only sets events, no more waiting
    filtering.median_3x3(tmpA, tmpB, tid, num_threads)
    # swap direction of filtering
    tmpA, tmpB = tmpB, tmpA
    events[tid, 1].set()

    for i in range(1, iterations):
        #Handle events
        if tid>0:
            events[tid-1, i-1].wait()
        if tid<num_threads-1:
            events[tid+1, i-1].wait()

        filtering.median_3x3(tmpA, tmpB, tid, num_threads)
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA
        events[tid, i].set()



if __name__ == '__main__':
    input_image = np.load('image.npz')['image'].astype(np.float32)

    t_record = [0]*16
    for n in range(1,17):
        with Timer() as t:
            new_image = py_median_3x3(input_image, 10, n)
        t_record[n-1] = t.interval

    print t_record
    plt.plot(range(1,17), t_record)
    plt.xlabel('number of threads')
    plt.ylabel('time')
    plt.show()

