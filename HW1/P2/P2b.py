from P2 import *
import random

# Your code here
sc = SparkContext(appName="P2b")
n = 2000
partitions = 100
rdd = sc.parallelize(xrange(n))
cartesian_rdd = rdd.cartesian(rdd).partitionBy(partitions, lambda x: random.randint(0, 99))
mandelbrot_rdd = cartesian_rdd.map(lambda (x, y): ((x, y), mandelbrot(y/500.0 - 2, x/500.0 - 2)))
draw_image(mandelbrot_rdd)
sums = sum_values_for_partitions(mandelbrot_rdd)
plt.hist(sums.collect())
plt.xlabel('Iterations')
plt.ylabel('Number of Partitions')
plt.show()