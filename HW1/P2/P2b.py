from P2 import *

from pyspark import SparkContext
import time
import random

sc = SparkContext(appName="P2", pyFiles=[])

in_list = [( (i, j), (i, j) ) for i in xrange(2000) for j in xrange(2000)]
random.shuffle(in_list)

rdd = sc.parallelize(in_list, 100)\
                    .mapValues( lambda (k,v): mandelbrot(v/500.0 - 2., k/500.0 - 2.) )


draw_image( rdd )
start_t = time.time()
effort = sum_values_for_partitions( rdd ).collect()
print 'Time =', time.time() - start_t
print effort

plt.figure()
plt.xlabel("Iterations")
plt.ylabel("Number of partitions")
plt.hist(effort)
plt.show()