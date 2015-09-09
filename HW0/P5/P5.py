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

def infiniteWorkersTime(N):
    time = 0
    n=0    
    if N < 2:
        return 0
    else:
        while N > 1:
            N = N / 2.0
            n+=1
        time = n
        return time    
        
single_worker = np.zeros(len(num_bags)) #initialize single worker vector
infinite_workers = np.zeros(len(num_bags)) 

for N in num_bags:
    single_worker[N] = singleWorkerTime(N) #fill in single worker vector

for N in num_bags:
    infinite_workers[N] = infiniteWorkersTime(N) #fill in single worker vector


plt.plot(num_bags,single_worker,'r-',label='1 worker')
plt.plot(num_bags,infinite_workers,'b-',label='infinite workers')
plt.legend()

