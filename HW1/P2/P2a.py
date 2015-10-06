from P2 import *

# Your code here
sc = SparkContext(appName="P2a")
n = 2000
rdd = sc.parallelize(xrange(n), 10)
cartesian_rdd = rdd.cartesian(rdd)
mandelbrot_rdd = cartesian_rdd.map(lambda (x, y): ((x, y), mandelbrot(y/500.0 - 2, x/500.0 - 2)))
draw_image(mandelbrot_rdd)
sums = sum_values_for_partitions(mandelbrot_rdd)
plt.hist(sums.collect())
plt.xlabel('Iterations')
plt.ylabel('Number of Partitions')
plt.show()