"""P2a.py"""

import findspark

#findspark.init()
#from pyspark import SparkContext
import pyspark

from P2 import *

import numpy as np
import matplotlib.pyplot as plt
# Your code here

#sc = pyspark.SparkContext("local", "P2a.py", pyFiles=['P2.py'] )
sc =pyspark.SparkContext()

data=[]

for i in range(0,1999):
  for j in range(0,1999):
    data.append([i,j])


rdd=sc.parallelize(data,100)
#print rdd.take(10)
rdd_mandelbrot=rdd.map(lambda (i,j):((i,j), mandelbrot((j/500)-2,(i/500)-2)))

rdd_sum=sum_values_for_partitions(rdd_mandelbrot)

print rdd_sum.take(10)

#draw_image(rdd_mandelbrot)
plt.hist(rdd_sum.collect(),bins=20)
plt.show()
#draw_image(rdd_sum)

