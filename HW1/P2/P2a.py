from P2 import *
import pyspark

# Your code here
sc = pyspark.SparkContext()

rdd_i = sc.parallelize(range(2000), 10)
rdd_j = sc.parallelize(range(2000), 10)

product_rdd = rdd_i.cartesian(rdd_j)
mandelbrot_rdd = product_rdd.map(lambda (i, j): ((i, j), mandelbrot(j/500.0 - 2, i/500.0 - 2)))

draw_image(mandelbrot_rdd)

sum_rdd = sum_values_for_partitions(mandelbrot_rdd)
plt.hist(sum_rdd.collect())
plt.savefig('P2a_hist.png')