from P2 import *

# My code here #

import pyspark
sc = pyspark.SparkContext()
para_sc = sc.parallelize(xrange(2000), 10)
cart = para_sc.cartesian(para_sc)
rdd = cart.map(lambda s: [s, mandelbrot((s[0]/500.0 - 2), (s[1]/500.0 - 2))])
draw_image(rdd)

rdd1 = sum_values_for_partitions(rdd)

plt.hist(rdd1.collect())
plt.savefig('P2a_hist.png')