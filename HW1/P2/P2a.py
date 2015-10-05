
# coding: utf-8

# In[1]:

#import findspark
#import os
#findspark.init('/home/chongmo/spark') # you need that before import pyspark.
import pyspark
from pyspark import SparkContext
sc =SparkContext()


# In[2]:

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm


# In[3]:

def mandelbrot(x, y):
    z = c = complex(x, y)
    iteration = 0
    max_iteration = 511  # arbitrary cutoff
    while abs(z) < 2 and iteration < max_iteration:
        z = z * z + c
        iteration += 1
    return iteration

def sum_values_for_partitions(rdd):
    'Returns (as an RDD) the sum of V for each partition of a (K, V) RDD'
    # note that the function passed to mapPartitions should return a sequence,
    # not a value.
    return rdd.mapPartitions(lambda part: [sum(V for K, V in part)])

def draw_image(rdd):
    '''Given a (K, V) RDD with K = (I, J) and V = count,
    display an image of count at each I, J'''

    data = rdd.collect()
    I = np.array([d[0][0] for d in data])
    J = np.array([d[0][1] for d in data])
    C = np.array([d[1] for d in data])
    im = np.zeros((I.max() + 1, J.max() + 1))
    im[I, J] = np.log(C + 1)  # log intensity makes it easier to see levels
    plt.imshow(im, cmap=cm.gray)
    plt.show()


# In[4]:

ii=sc.parallelize(range(2000), numSlices=10)
jj=sc.parallelize(range(2000), numSlices=10)
ij=ii.cartesian(jj)


# In[5]:

iterations=(ij.map(lambda (i,j):((i,j), mandelbrot(j/500.0-2, i/500.0-2)))).persist()


# In[6]:

#draw_image(iterations)


# In[22]:

hist=sum_values_for_partitions(iterations)


# In[24]:

plt.hist(hist.collect())
plt.show()



