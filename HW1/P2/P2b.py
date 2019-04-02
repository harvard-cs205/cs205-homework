from P2 import *

# My code here

import pyspark
import random
sc = pyspark.SparkContext()
## shuffle 
x_ = range(2000)
random.shuffle(x_)
x = sc.parallelize(x_, 10)
cart = x.cartesian(x)
rdd = cart.map(lambda s: [s, mandelbrot((s[0]/500.0 - 2), (s[1]/500.0 - 2))])
draw_image(rdd)
rdd1 = sum_values_for_partitions(rdd)

plt.hist(rdd1.collect())
plt.savefig('P2b_hist.png')

