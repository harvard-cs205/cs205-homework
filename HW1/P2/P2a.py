
# coding: utf-8

# In[1]:

import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext(appName="Spark 2")


# In[2]:

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm


# In[3]:

# Given functions
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


# In[77]:

# Construct indices to span the 2000 by 2000 grid 
# of Mandelbrot computations
gridsz = 2000
n_parts = 100
rdd = sc.parallelize(np.arange(gridsz), n_parts)
rdd_paired = rdd.cartesian(rdd)


# In[78]:

# Compute mendelbrot on each pixel. 
# Note that dimensions (x, y) correspond to (j, i)
def map2xy(ij_coord):
    return ij_coord / 250.0 - 2
res = rdd_pair.map(lambda IJ: (IJ, mandelbrot(map2xy(IJ[1]), map2xy(IJ[0])))) 


# In[79]:

draw_image(res)


# In[86]:

# Plot and save the histogram of work across partitions
plt.hist(sum_values_for_partitions(res).collect(), 
         facecolor = 'k', edgecolor = 'None')
plt.xlabel('# iterations')
plt.ylabel('# partitions')
plt.savefig('P2a_hist.png')

