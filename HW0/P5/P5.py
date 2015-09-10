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
        print "elif"
        return infTime(N,t)
    else:
        N = N-1
        N = N / 2.0
        N += 1
        t += 1
        print "else"
        return infTime(N,t)

        
single_worker = np.zeros(len(num_bags)) #initialize single worker vector
infinite_workers = np.zeros(len(num_bags)) #initialize infinite worker vector

for N in num_bags:
    single_worker[N] = singleWorkerTime(N) #fill in single worker vector

for N in num_bags:
    infinite_workers[N] = infTime(N,0) #fill in single worker vector

plt.plot(num_bags,single_worker,'r-',label='1 worker')
plt.plot(num_bags,infinite_workers,'b-',label='infinite workers')
plt.legend()

