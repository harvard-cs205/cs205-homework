from P2 import *

from pyspark import SparkContext
import time
#
sc = SparkContext(appName="P2", pyFiles=[])

rdd = sc.parallelize(xrange(2000))
rdd = rdd.cartesian(rdd)
rdd = rdd.partitionBy(100).map( lambda (k,v): ( (k,v), mandelbrot(v/500.0 - 2.,k/500.0 - 2.) ) )

draw_image( rdd )

start_t = time.time()
effort = sum_values_for_partitions( rdd ).collect()
print 'Time =', time.time() - start_t
print effort

print effort
plt.figure()
plt.xlabel("Iterations")
plt.ylabel("Number of partitions")
plt.hist(effort)
plt.show()