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

def py_median_3x3_nt(image, iterations=10):
    ''' repeatedly filter with a 3x3 median '''
    # THIS CODE IS THE UNTHREADED PART. MEANING IT HAS NOT BEEN 
    # CHANGED FROM THE ORIGINAL VERSION
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    for i in range(iterations):
        filtering.median_3x3(tmpA, tmpB, 0, 1)
        # swap direction of filtering
        tmpA, tmpB = tmpB, tmpA

    return tmpA

def py_median_3x3_t(image, iterations, num_threads):
    ''' repeatedly filter with a 3x3 median '''
    # IN THIS VERSION THREADING IS IMPLEMENTED SUCH THAT 
    # EACH PIXEL ROW MUST WAIT ON ITS NEIGHBOURING ROWS
    # TO BE COMPLETED BEFORE CONTINUING WITH THE COMPUTATION
    tmpA = image.copy()
    tmpB = np.empty_like(tmpA)

    # INITIALIZE A NUMPY ARRAY OF EVENTS WITH ENOUGH EVENTS FOR EACH
    # THREAD IN EACH ITTERATION. THIS ARRAY CONTAINS ONE ROW
    # FOR EACH THREAD AND ONE COLUMN FOR EACH ITERATION
    threadEvent = np.array([[threading.Event() for ii in range(iterations)] for jj in range(num_threads)])

    # NOW TO LOOP OVER THE NUMBER OF THREADS AND ASSIGN EACH THREAD A SPECIFIC JOB TO COMPLETE
    # WASED ON THE ASSIGN THREAD FUNCTION
    for threadNumber in range(num_threads):
        # CREATE THE ARGUMENTS TO BE PASSED INTO THE THREAD TARGET FUNCTION FOR EACH INDIVIDUAL THREAD
        argumentsPassed = (iterations, threadNumber, num_threads, threadEvent, tmpA, tmpB)

        # GENERATING THE THREAD AND ASSIGN THE THREAD THE WORK TO BE DONE WITH THE ARGUMENTS 
        # DEFINED ABOVE
        thread = threading.Thread(target = assign_thread, args=argumentsPassed)

        # START RUNNIGN THE THREAD ASSINED BY THREADNUMBER
        thread.start()

    # WAIT FOR ALL OF THE THREADS TO FINISH AND GET THEMTOGTHER TO OBTAIN THE FINIAL VALUE OF
    # THE IMAGE AT THE END OF THE COMPUTATION.
    thread.join()

    # RETURN THE IMAGE BACK TO BE ASSIGNED TO THE NEW IMAGE
    return tmpA

def assign_thread(iterations, threadNumber, num_threads, threadEvent, temp1, temp2):
    # THIS METHOD DOES NOT WORK FOR THREAD LESS THAN 2, FOR A THREAD OF 1 YOU MUST 
    # USE THE NON-THREDED VERSION OF THE MEDIAN 3X3
    # GO OVER THE CODE THE ITTERATIONS FORE EACH THREAD
    for iteration in range(iterations):
        # CHECK TO SEE WHICH THREAD IT IS AND WHAT ITTERATION IT IS
        # WE MUST MAKE SURE THAT IT IS NOT ITTERATION 0 SO THAT 
        # IT DOES NOT LOCK. IF ITTERATION IS 0 WE CAN PROCEED WITHOUG
        # THREAD ALLOCATION
        if iteration ==0:
            pass
        
        # FOR THREAD 0 WE MUST MAKE SURE THAT THE EVENTS AROUND IT ARE DONE
        elif threadNumber ==0:
            # WAIT UNTIL THE NEIGHBOURING THREAD IS DONE IN THE PREVISOUS
            # ITTRATION. WE WANT TO MAKE SURE THAT THE IMAGE IS READY ALONG 
            # THE NEIGHBORING LINES BEFORE THE THREAD CAN CONTINUE TO DO WORK
            threadEvent[threadNumber + 1,iteration - 1].wait()
            threadEvent[num_threads-1,iteration - 1].wait()
        
        # THIS LOOKS AT WHAT THREAD AND IF IT IS THE LAST THREAD IT MUSY ONLY WAIT FOR THE ONE 
        # THREAD THAT COMES BEFORE. THIS ESURES THAT THE DATA IN ITS NEIGHBOR IS READY FOR IT 
        # TO PERFORM ITS FILTER. YOU MUST CHECK THE ONE BEFORE AND THE FIRST THREAD WHICH WILL BE
        # THE NEIGHBOURING THREAD
        elif threadNumber==num_threads-1:
            threadEvent[threadNumber -1 , iteration-1].wait()
            threadEvent[0 , iteration-1].wait()

        # FOR ALL OTHER THREADS WE MUST CHECK BOTH NEIGHBOURING THREADS WORK, THIS MAKES SURE THAT 
        # THE ALL OF THE DATA IS READY FOR ANALYSIS
        else: 
            threadEvent[threadNumber-1, iteration-1].wait()
            threadEvent[threadNumber+1, iteration-1].wait()

        # THIS PERFROMES THE FILTERING FOR THE RESPECTIVE THREADS. BEFORE PERFOMING THEIR COMPUTATIONS
        # EACH THREAD GOES THROUGH THE LOGIC ABOVE TO ENSURES THAT THEY ARE ACCECCING DATA THAT HAS
        # ALREADY BEEN WORKED ON ON THE PREVIOUS ITTERATION 
        filtering.median_3x3(temp1, temp2, threadNumber, num_threads)
        threadEvent[threadNumber, iteration].set() #tell the event that this thread is complete
        temp1, temp2 = temp2, temp1


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

    # verify correctness for non-threaded
    from_cython = py_median_3x3_nt(input_image, 2)
    from_numpy = numpy_median(input_image, 2)
    assert np.all(from_cython == from_numpy)

    # TIMING AND PRINTING FOR NON-THREADED
    with Timer() as t:
        new_image = py_median_3x3_nt(input_image, 10)
    print("For non threaded it takes {} seconds for 10 filter passes.".format(t.interval))

    # verify correctness for 2 threads
    from_cython = py_median_3x3_t(input_image, 10, 2)
    from_numpy = numpy_median(input_image, 10)
    assert np.all(from_cython == from_numpy)

    # TIMING AND PRINTING FOR 2 THREADS
    with Timer() as t:
        new_image = py_median_3x3_t(input_image, 10, 2)
    print("For 2 threads it takes {} seconds for 10 filter passes.".format(t.interval))
    
    # verify correctness for 4 threads
    from_cython = py_median_3x3_t(input_image, 10, 4)
    from_numpy = numpy_median(input_image, 10)
    assert np.all(from_cython == from_numpy)

    # TIMING AND PRINTING FOR 4 THREADS
    with Timer() as t:
        new_image = py_median_3x3_t(input_image, 10, 4)
    print("For 4 threads it takes {} seconds for 10 filter passes.".format(t.interval))

    pylab.figure()
    pylab.imshow(new_image[1200:1800, 3000:3500])
    pylab.title('after - zoom')

    
    pylab.show()
