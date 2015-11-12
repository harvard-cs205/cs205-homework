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

    # list of threads
    thread_list = [0 for i in range(num_threads)]
    # list of (iteration, thread) events
    iteration_event_list = [[threading.Event() for i in range(num_threads)] for j in range(iterations)]

    # # list of events for even-numbered iterations
    # even_iteration_events_list = [threading.Event() for i in range(num_threads)]
    # # list of events for odd-numbered iterations
    # odd__iteration_events_list = [threading.Event() for i in range(num_threads)]
    # # list that makes it easy to access other lists
    # even_odd_list = [even_iteration_events_list, odd__iteration_events_list]

    # given the index of a thread, returns a list of the indexes of threads
    # that need the result of this thread's previous iteration
    # def get_neighbors(thread_index):
    #     if num_threads == 1:
    #         return []
    #     if thread_index == 0:
    #         return [1]
    #     if thread_index == num_threads-1:
    #         return [num_threads-2]
    #     return [thread_index-1, thread_index+1]

    # def threadFilter(thread_index, tmpA, tmpB):
    #     for current_iteration in range(iterations):
    #         # if this is the first iteration or there is only one thread,
    #         # we wouldn't have to wait for anything
    #         if current_iteration > 0 and num_threads >= 2:
    #             # check if thread is 0 so that we don't check thread-1
    #             if thread_index == 0:
    #                 even_odd_list[(current_iteration-1)%2][1].wait()
    #             # check if last thread so that we don't go over bound of threads list
    #             elif thread_index == num_threads-1:
    #                 even_odd_list[(current_iteration-1)%2][num_threads-2].wait()
    #             else:
    #                 # wait for thread-1 to be done with iteration-1
    #                 even_odd_list[(current_iteration-1)%2][thread_index-1].wait()
    #                 # wait for thread+1 to be done with iteration+1
    #                 even_odd_list[(current_iteration-1)%2][thread_index+1].wait()
    #         # run function with thread_index as first row and step of num_threads
    #         filtering.median_3x3(tmpA, tmpB, thread_index, num_threads)
    #         # swap buffers
    #         tmpA, tmpB = tmpB, tmpA
    #         # fire current event
    #         even_odd_list[current_iteration%2][thread_index].set()
    #         # if the other threads are done with this thread's previous iteration,
    #         # clear that iteration, index event so that it can be used in the future
    #         neighbors_done_with_previous_iteration = True
    #         for neighbor in get_neighbors(thread_index):
    #             if not even_odd_list[current_iteration%2][neighbor].is_set():
    #                 neighbors_done_with_previous_iteration = False
    #                 break
    #         if neighbors_done_with_previous_iteration:
    #             even_odd_list[(current_iteration-1)%2][thread_index].clear()

    def threadFilter(thread_index, tmpA, tmpB):
        for current_iteration in range(iterations):
            # if this is the first iteration or there is only one thread,
            # we wouldn't have to wait for anything
            if current_iteration > 0 and num_threads >= 2:
                # check if thread is 0 so that we don't check thread-1
                if thread_index == 0:
                    iteration_event_list[current_iteration-1][1].wait()
                # check if last thread so that we don't go over bound of threads list
                elif thread_index == num_threads-1:
                    iteration_event_list[current_iteration-1][num_threads-2].wait()
                else:
                    # wait for thread-1 to be done with iteration-1
                    iteration_event_list[current_iteration-1][thread_index-1].wait()
                    # wait for thread+1 to be done with iteration+1
                    iteration_event_list[current_iteration-1][thread_index+1].wait()
            # run function with thread_index as first row and step of num_threads
            filtering.median_3x3(tmpA, tmpB, thread_index, num_threads)
            # swap buffers
            tmpA, tmpB = tmpB, tmpA
            # fire current event
            iteration_event_list[current_iteration][thread_index].set()
            
    # create and start threads
    for thread_index in range(num_threads):
        thread_list[thread_index] = threading.Thread(target=threadFilter, args=(thread_index, tmpA, tmpB))
        thread_list[thread_index].start()
    # wait for all threads to be done
    for thread in thread_list:
        thread.join()
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
        new_image = py_median_3x3(input_image, 10, 4)

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    print("{} seconds for 10 filter passes.".format(t.interval))
    pylab.show()
