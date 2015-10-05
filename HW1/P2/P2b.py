
# coding: utf-8

# In[1]:

import findspark
import os
import random
findspark.init('/home/chongmo/spark') # you need that before import pyspark.
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


# In[4]: code for P2b2_hist.png

x=range(2000)
random.shuffle(x)
ii=sc.parallelize(x, numSlices=10)
random.shuffle(x)
jj=sc.parallelize(x, numSlices=10)
ij=ii.cartesian(jj)


# In[9]: code for P2b1_hist.png

#import random
#x=range(2000)
#ii=sc.parallelize(x, numSlices=10)
#jj=sc.parallelize(x, numSlices=10)
#ij=ii.cartesian(jj)
#ij=ij.partitionBy(numPartitions=100, partitionFunc=lambda x: random.randint(0, 100))


# In[10]:

iterations=(ij.map(lambda (i,j):((i,j), mandelbrot(j/500.0-2, i/500.0-2)))).persist()


# In[11]:

#draw_image(iterations)
hist=sum_values_for_partitions(iterations)


# In[12]:

plt.hist(hist.collect())
plt.show()


# In[ ]:



