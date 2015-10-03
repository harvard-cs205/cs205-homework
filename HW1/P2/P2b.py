from P2 import *

import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext()

partitions2 = sc.parallelize(tuples).map(lambda xy: (xy, mandelbrot((xy[1]/500.0)-2,(xy[0]/500.0)-2))).partitionBy(100, lambda xy: random.randrange(0, 100, 1))
