from P2 import *

import findspark
findspark.init('/Users/george/Documents/spark-1.5.0')
import pyspark
import matplotlib.pyplot as plt
import random
 
sc = pyspark.SparkContext()

# create rdd
rdd = sc.parallelize(xrange(2000), 10)
rdd = rdd.cartesian(rdd)

# partition randomly
rdd = rdd.partitionBy(100, lambda x: random.randint(0, 99))

# use formula provided by pset
rdd_map = rdd.map(lambda (i,j): ((i,j), (j/500.0-2, i/500.0-2)))

# apply mandelbrot
result = rdd_map.mapValues(lambda (x,y): mandelbrot(x,y))

draw_image(result)

# calculate partition work
answer = sum_values_for_partitions(result).collect()

# plot histogram
plt.hist(answer, bins=20)
plt.title("Histogram of Per-Partition Work")
plt.ylabel("Iterations")
plt.xlabel("Partitions")
plt.show()
