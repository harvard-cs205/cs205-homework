
# Your code here
import pyspark
from pyspark import SparkContext
sc = SparkContext()
from P2 import *
 
# construct 2000*2000 matrix and make 100 partitions
rdd1 = sc.parallelize(xrange(1,2001,1), 10)
rdd2 = sc.parallelize(xrange(1,2001,1), 10)
cart = rdd1.cartesian(rdd2)

# calculate iterations on each pixel 
rdd3 = cart.map(lambda r: [r, mandelbrot(r[1]/500.0-2, r[0]/500.0-2)]).persist()

# draw image and histogram
draw_image(rdd3)
value = sum_values_for_partitions(rdd3)
plt.hist(value.collect())
plt.savefig('P2a_hist.png')






