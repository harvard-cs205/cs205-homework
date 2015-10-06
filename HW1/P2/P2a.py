
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
#Clean Version
arr = xrange(2000)
rdd_single = sc.parallelize(arr,10)
rdd = rdd_single.cartesian(rdd_single)

iters = rdd.map(lambda (i,j): ((i,j), mandelbrot(j / 500.0 - 2, i / 500.0 - 2)))
num_per_part = sum_values_for_partitions(iters)
draw_image(iters)
num_per_part_list = num_per_part.collect()



# In[9]:

#plot histogram
plt.hist(num_per_part_list, color="black")
plt.title("Compute per Partition")
plt.xlabel("Compute (Effort)")
plt.ylabel("Frequency")
plt.savefig("P2a_hist.png")


# In[ ]:



