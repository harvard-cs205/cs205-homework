from P2 import *

import pyspark

from P2 import *

import numpy as np
import matplotlib.pyplot as plt

#sc = pyspark.SparkContext("local", "P2a.py", pyFiles=['P2.py'] )
sc =pyspark.SparkContext()

data=[]
for i in range(0,1999):
    for j in range(0,1999):
      data.append([i,j])

#print data
rdd=sc.parallelize(data)
rdd=rdd.repartition(100)

rdd_mandelbrot=rdd.map(lambda (i,j):((i,j), mandelbrot((j/500.0)-2,(i/500.0)-2)))

rdd_sum=sum_values_for_partitions(rdd_mandelbrot)

plt.hist(rdd_sum.collect(),bins=20)
plt.show()
