from P2 import *

import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext()

tuples = []
for i in xrange(2000):
    for j in xrange(2000):
        tuples.append((i,j))

partitions = sc.parallelize(tuples, 100).map(lambda xy: (xy, mandelbrot((xy[1]/500.0)-2,(xy[0]/500.0)-2)))

draw_image(partitions)
