from P2 import *
from pyspark import SparkContext
import random

# Your code here

def partitionNode(node):
    x,y = node
    min_x = 800
    max_x = 1000
    min_y = 700
    max_y = 1200
    if x >= min_x and x < max_x and y >= min_y and y < max_y:
        return (y-min_y)/5
    else:
        return random.randint(0, 99)

sc = SparkContext("local", "P2a")
rdd = sc.parallelize(xrange(0, 2000), 10)
cross = rdd.cartesian(rdd).map(lambda x: (x,0)).partitionBy(100, partitionNode)
map_mandelbrot = cross.map(lambda x: (x[0],mandelbrot(x[0][0]/500.0 - 2, x[0][1]/500.0 - 2)))
sum_partitions = sum_values_for_partitions(map_mandelbrot)
#print sum_partitions.take(100)
#draw_image(map_mandelbrot)
sum_array = sum_partitions.collect()
plt.hist(sum_array)
