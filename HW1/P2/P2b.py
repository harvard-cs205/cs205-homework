
# coding: utf-8

# In[1]:

import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext()

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import seaborn
from P2 import *


# In[80]:

# Construct indices to span the 2000 by 2000 grid 
# of Mandelbrot computations
gridsz = 2000
n_parts = 10
rdd = sc.parallelize(np.arange(gridsz), n_parts)

# This should have n_parts^2 partitions
rdd_paired = rdd.cartesian(rdd).map(lambda x: (x, 0))


# In[82]:

# Check the number of partitions
print(len(rdd.glom().collect()))
print(len(rdd_paired.glom().collect()))


# In[88]:

# Hash to a random partition
def rhash(x):
    return np.random.random_integers(0, 99)
rdd_custom_rand = rdd_paired.partitionBy(n_parts**2, rhash)


# In[92]:

# Compute mendelbrot on each pixel. 
# Note that dimensions (x, y) correspond to (j, i)
def map2xy(ij_coord):
    return ij_coord / (gridsz / 4.0) - 2
res_rand = rdd_custom_rand.map(lambda IJ: (IJ[0], mandelbrot(map2xy(IJ[0][1]), map2xy(IJ[0][0])))) 


# In[21]:

draw_image(res)


# In[94]:

work_load_rand = sum_values_for_partitions(res_rand).collect()


# In[106]:

# Plot and save the histogram of work across partitions
plt.figure()
plt.hist(work_load_rand, facecolor = 'k', edgecolor = 'w')
plt.xlabel('# iterations')
plt.ylabel('# partitions')
plt.title('Load distribution: random partition assignment')
# plt.show()
plt.savefig('P2b_hist.png')

