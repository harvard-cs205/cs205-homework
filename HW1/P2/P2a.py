
# coding: utf-8

# In[1]:

import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext(appName="Spark 2")

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import seaborn
from P2 import *


# In[4]:

# Construct indices to span the 2000 by 2000 grid 
# of Mandelbrot computations
gridsz = 2000
n_parts = 10
rdd = sc.parallelize(np.arange(gridsz), n_parts)
rdd_paired = rdd.cartesian(rdd)


# In[7]:

# Compute mendelbrot on each pixel. 
# Note that dimensions (x, y) correspond to (j, i)
def map2xy(ij_coord):
    return ij_coord / (gridsz / 4.0) - 2
res = rdd_paired.map(
    lambda IJ: (IJ, mandelbrot(map2xy(IJ[1]), map2xy(IJ[0])))).cache()


# In[8]:

draw_image(res)


# In[10]:

sum_vals = sum_values_for_partitions(res).collect()


# In[14]:

# Plot and save the histogram of work across partitions
plt.figure()
plt.hist(sum_vals, facecolor = 'k', edgecolor = 'w')
plt.xlabel('# iterations')
plt.ylabel('# partitions')
plt.title('Load distribution: default partition assignment')
# plt.show()
plt.savefig('P2a_hist.png')

