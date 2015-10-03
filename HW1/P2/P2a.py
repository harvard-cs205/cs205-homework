from P2 import *

# Your code here
import pyspark
sc = pyspark.SparkContext(appName="Spark1")
import numpy as np 


xs = sc.parallelize(range(2000), 10)
grid = xs.cartesian(xs)
#grid.take(10)
#xys = grid.map(lambda (i, j): ((i, j), (j/500.0 - 2, i/500.0 - 2))).repartition(100)
xys = grid.map(lambda (i, j): ((i, j), (j/500.0 - 2, i/500.0 - 2)))
reses = xys.mapValues(lambda (x, y): mandelbrot(x, y))
#print xys.take(10)
#print reses.take(10)
#draw_image(reses)
distribution = sum_values_for_partitions(reses).collect()
plt.figure()
plt.hist(distribution)
plt.show()


