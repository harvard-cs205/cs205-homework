from P2 import *
from pyspark import SparkContext

# Your code here
sc = SparkContext("local", "P2a")
rdd = sc.parallelize(xrange(0, 2000), 10)
cross = rdd.cartesian(rdd)
map_mandelbrot = cross.map(lambda x: (x,mandelbrot(x[0]/500.0 - 2, x[1]/500.0 - 2)))
sum_partitions = sum_values_for_partitions(map_mandelbrot)
#print sum_partitions.take(100)
#draw_image(map_mandelbrot)
sum_array = sum_partitions.collect()
plt.hist(sum_array)
