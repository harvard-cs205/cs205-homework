from P2 import *
import pyspark
import random

# Your code here
sc = pyspark.SparkContext()

list_i = range(2000)
list_j = range(2000)
random.shuffle(list_i)
random.shuffle(list_j)
rdd_i = sc.parallelize(list_i, 10)
rdd_j = sc.parallelize(list_j, 10)


product_rdd = rdd_i.cartesian(rdd_j)
mandelbrot_rdd = product_rdd.map(lambda (i, j): ((i, j), mandelbrot(j/500.0 - 2, i/500.0 - 2)))

draw_image(mandelbrot_rdd)

sum_rdd = sum_values_for_partitions(mandelbrot_rdd)
plt.hist(sum_rdd.collect())
plt.savefig('P2b_hist.png')