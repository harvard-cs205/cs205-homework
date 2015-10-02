
# coding: utf-8

# In[1]:

# github.com/minrk/findspark
import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext(appName="Spark1")

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from P2 import mandelbrot, sum_values_for_partitions, draw_image
'''
#RePartitioning
arr = xrange(2000)
rdd_single = sc.parallelize(arr,10)
rdd = rdd_single.cartesian(rdd_single)
rdd = rdd.repartition(100)
iters = rdd.map(lambda (i,j): ((i,j), mandelbrot(j / 500.0 - 2, i / 500.0 - 2)))
#print iters.take(10)
num_per_part = sum_values_for_partitions(iters)
#draw_image(iters)
plt.hist(num_per_part.collect())
#MAKE HISTOGRAM PRETTIER
plt.savefig("P2c_hist.png")
'''
#Improved Partitioning
arr = xrange(2000)
rdd_single = sc.parallelize(arr,10)
rdd = rdd_single.cartesian(rdd_single)
rdd = rdd.partitionBy(100)
iters = rdd.map(lambda (i,j): ((i,j), mandelbrot(j / 500.0 - 2, i / 500.0 - 2)))
#print iters.take(10)
num_per_part = sum_values_for_partitions(iters)
#draw_image(iters)
plt.hist(num_per_part.collect())
#MAKE HISTOGRAM PRETTIER
plt.savefig("P2b_hist.png")


# In[ ]:



