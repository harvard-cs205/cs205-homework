# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

bags=256 #number of bags to count
workers = 8 #number of workers

def time(bags,workers):
    if bags <=1:
        return 0
    else:
        return bags / workers - 1

i=0 #initialize loop
time_array = np.zeros(bags+1)
while i<=bags:    
    time_array[i] = time(i,workers)
    i+=1

x=np.linspace(0,bags,bags+1)
plt.plot(x,time_array,'--b', label='infinite workers')

c = 1 #number of workers

i=2 #initialize loop
time = np.zeros(N+1)
time[0]=0
time[1]=0
while i<=N:    
    time[i] = timeToCountBags(i,c)
    i+=1

plt.plot(x,time,label='1 workers')
plt.legend()

plt.show()




#def timeToCountBags(num_bags,num_workers):
 #   return num_bags/num_workers - 1