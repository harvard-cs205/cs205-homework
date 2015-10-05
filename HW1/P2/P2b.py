from P2 import *

# Your code here
import pyspark
sc = pyspark.SparkContext(appName="Spark1")
import numpy as np 

# Initialize the RDD on 10 partitions so that after the cartesian product, it is 100
xs = sc.parallelize(range(2000), 10)
grid = xs.cartesian(xs)
# Use a partitionBy so that items are spread out!
xys = grid.map(lambda (i, j): ((i, j), (j/500.0 - 2, i/500.0 - 2))).partitionBy(100, lambda (i, j): 10*(i%10) + j%10)
reses = xys.mapValues(lambda (x, y): mandelbrot(x, y))

draw_image(reses)
distribution = sum_values_for_partitions(reses).collect()
plt.figure()
plt.hist(distribution)
plt.show()
