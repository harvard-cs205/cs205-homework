# Your code here

from P2 import *

# Your code here

import findspark
findspark.init('/home/zelong/spark')
import pyspark
import random

sc = pyspark.SparkContext()
i = range(2000)
j = range(2000)

random.shuffle(i)
random.shuffle(j)

rdd_i = sc.parallelize(i, 10)
rdd_j = sc.parallelize(j, 10)

rdd_ij = rdd_i.cartesian(rdd_j)
rdd_iter = rdd_ij.map(lambda j: (j, mandelbrot ( (j[1]/500.0 ) - 2 , (j[0]/500.0) - 2)))

rdd_sum = sum_values_for_partitions(rdd_iter)
temp = rdd_sum.collect()

#draw_image(rdd_iter)
plt.hist(temp)
plt.savefig("P2b_hist")
