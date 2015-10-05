import pyspark
from pyspark import SparkContext
sc = SparkContext()
from P2 import *
import random 

# make two ranges of 2000 numbers and random shuffle on both
x = range(2000)
y = range(2000)
random.shuffle(x)
random.shuffle(y)

# make 100 partitions
rdd1 = sc.parallelize(x, 10)
rdd2 = sc.parallelize(y, 10)
cart = rdd1.cartesian(rdd2)

# calculate iterations on each pixel
new_rdd = cart.map(lambda r: [r, mandelbrot(r[1]/500.0-2, r[0]/500.0-2)]).persist()

# draw image and histogram
draw_image(new_rdd)
value = sum_values_for_partitions(new_rdd)
plt.hist(value.collect())
plt.savefig('P2b_hist.png')
