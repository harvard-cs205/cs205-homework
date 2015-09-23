"""P2a.py"""

import findspark

findspark.init()

#from pyspark import SparkContext
import pyspark

from P2 import *

import numpy as np

# Your code here

sc = pyspark.SparkContext("local", "P2a.py", pyFiles=['P2.py'] )

data=np.zeros([2000,2000,2])

for i in range(1,2000):
  for j in range(1,2000):
    data[i,j,0]=(j/500.0)-2
    data[i,j,1]=(i/500.0)-2

rdd=sc.parallelize(data,100)


rdd.map(mandelbrot)

rdd.take(10)
