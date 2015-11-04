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

    thread_list = [0 for i in range(num_threads)]
    even_iteration_events_list = [threading.Event() for i in range(num_threads)]
    odd__iteration_events_list = [threading.Event() for i in range(num_threads)]
    even_odd_list = [even_iteration_events_list, odd__iteration_events_list]

    
    def get_neighbors(thread_index):
        if num_threads == 1:
            return []
        if thread_index == 0:
            return [1]
        if thread_index == num_threads-1:
            return [num_threads-2]
        return [thread_index-1, thread_index+1]

    def threadFilter(thread_index, tmpA, tmpB):
        for current_iteration in range(iterations):
            if current_iteration > 0 and num_threads >= 2:
                if thread_index == 0:
                    even_odd_list[(current_iteration-1)%2][1].wait()
                elif thread_index == num_threads-1:
                    even_odd_list[(current_iteration-1)%2][num_threads-2].wait()
                else:
                    even_odd_list[(current_iteration-1)%2][thread_index-1].wait()
                    even_odd_list[(current_iteration-1)%2][thread_index+1].wait()
            filtering.median_3x3(tmpA, tmpB, thread_index, num_threads)
            tmpA, tmpB = tmpB, tmpA
            even_odd_list[current_iteration%2][thread_index].set()
            neighbors_done_with_previous_iteration = True
            for neighbor in get_neighbors(thread_index):
                if not even_odd_list[current_iteration%2][neighbor].is_set():
                    neighbors_done_with_previous_iteration = False
                    break
            if neighbors_done_with_previous_iteration:
                even_odd_list[(current_iteration-1)%2][thread_index].clear()



    for thread_index in range(num_threads):
        thread_list[thread_index] = threading.Thread(target=threadFilter, args=(thread_index, tmpA, tmpB))
        thread_list[thread_index].start()
    for thread in thread_list:
        thread.join()
    return tmpA



    # for i in range(iterations):
    #     filtering.median_3x3(tmpA, tmpB, 0, 1)
    #     # swap direction of filtering
    #     tmpA, tmpB = tmpB, tmpA

    # return tmpA

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
