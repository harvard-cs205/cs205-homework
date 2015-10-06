from P2 import *

# Your code here
import findspark
findspark.init('/home/xiaowen/spark')
import pyspark

# build sparkcontext object
sc = pyspark.SparkContext(appName="P2")

# initialize 2000*2000 pixels
rdd_1d = sc.parallelize(range(2000), 10)
rdd_2d = rdd_1d.cartesian(rdd_1d)

# (K,V) for draw_image function where K = (I, J) and V = count
rdd_kv = rdd_2d.map(lambda x: (x, mandelbrot((x[1]/500.0)-2, (x[0]/500.0)-2)))

# # draw image
# draw_image(rdd_kv)

# draw histogram of compute "effort"
rdd_sum = sum_values_for_partitions(rdd_kv)
plt.hist(rdd_sum.collect())
plt.savefig("/home/xiaowen/cs205-homework/HW1/P2/P2a_hist.png")
