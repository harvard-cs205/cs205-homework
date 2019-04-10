from P2 import *

# Your code here
import pyspark
from pyspark import SparkContext, SparkConf
import random
sc = SparkContext()

xx = yy = range(2000)

# shuffle
random.shuffle(xx)
random.shuffle(yy)

rdd_xx = sc.parallelize(xx, 10)
rdd_yy = sc.parallelize(yy, 10)

rdd_xy = rdd_xx.cartesian(rdd_yy)
res = sum_values_for_partitions(rdd_xy.map(lambda x: (x, mandelbrot ( (x[1]/500.0 ) - 2 , (x[0]/500.0) - 2)))).collect()

plt.hist(res)
plt.savefig("P2b_hist")
