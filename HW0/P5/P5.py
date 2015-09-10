# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

num_bags = range(0,1000) #initialize number of bags array
#represnted like 0,1,2,3,4,5,6 bags...

def singleWorkerTime(N): #create function that calculates time to count n bags by single worker
    if N < 2:
        return 0
    else:
        return N-1

def infTime(N,t): #wrote a recursive formula to calculate number of seconds it would take infinite workers to count N bags
    if  N < 2:
        return t
    elif N == 2:
        t += 1
        return t
    elif N % 2 == 0:
        N = N / 2.0
        t += 1 
        return infTime(N,t)
    else:
        N = N-1
        N = N / 2.0
        N += 1
        t += 1
        return infTime(N,t)

        
single_worker = np.zeros(len(num_bags)) #initialize single worker vector
infinite_workers = np.zeros(len(num_bags)) #initialize infinite worker vector

for N in num_bags:
    single_worker[N] = singleWorkerTime(N) #fill in single worker vector

for N in num_bags:
    infinite_workers[N] = infTime(N,0) #fill in single worker vector

plt.plot(num_bags,single_worker,'r-',label='1 worker')
plt.plot(num_bags,infinite_workers,'b-',label='infinite workers')
plt.xlabel('Number of Bags (N)')
plt.ylabel('Time (Seconds)')
plt.legend()

#tried writing a function that takes communication and bag distribution into account. realized it was no longer needed when i read hte prompt further.
"""def infTimeOneBag(N,t): #wrote a recursive formula to calculate number of seconds it would take infinite workers to count N bags
    if  N < 2:
        return t
    elif N == 2 & t <= 2:
        t += 1 #assumes there was only 1 cashier and no need to communicate with others. so only 1  second to count the 2 bags individually
        return t
    elif N == 2:
        t+=2 #2 for communication and for the calculation
        return t
    elif N % 2 == 0 & t <= N:
        N = N / 2.0
        t += 1 #no communication first step
        return infTimeOneBag(N,t)
    elif N % 2 == 0 & t > N:
        N = N / 2.0
        t += 2 #communication here between cashiers
        return infTimeOneBag(N,t)
    elif N % 2 != 0 & t <= N:
        N = N-1
        N = N / 2.0
        N += 1
        t += 1 #no communication in first step
        return infTimeOneBag(N,t)
    else:
        N = N-1
        N = N / 2.0
        N += 1
        t += 2 #communication here between cashiers
        return infTimeOneBag(N,t)

print infTimeOneBag(0,0)
print infTimeOneBag(1,1)
print infTimeOneBag(2,2)
print infTimeOneBag(3,3)
print infTimeOneBag(4,4)
print infTimeOneBag(5,5)
print infTimeOneBag(256,256)
print infTimeOneBag(256,128)
print singleWorkerTime(256)"""