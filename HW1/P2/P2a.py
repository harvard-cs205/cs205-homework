
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

# In[3]:
from P2 import mandelbrot, sum_values_for_partitions, draw_image
#Clean Version
arr = xrange(2000)
rdd_single = sc.parallelize(arr,10)
rdd = rdd_single.cartesian(rdd_single)
#print dir(rdd)

iters = rdd.map(lambda (i,j): ((i,j), mandelbrot(j / 500.0 - 2, i / 500.0 - 2)))
num_per_part = sum_values_for_partitions(iters)
#draw_image(iters)
plt.hist(num_per_part.collect())
#MAKE HISTOGRAM PRETTIER
plt.savefig("P2a_hist.png")

