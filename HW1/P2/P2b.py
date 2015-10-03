from P2 import *

# Your code here
import pyspark
from pyspark import SparkContext, SparkConf
import random
sc = SparkContext()

my_x = range(0,2000,1)
my_y = range(0,2000,1)
random.shuffle(my_x)
random.shuffle(my_y)

x = sc.parallelize(my_x, 10)
y = sc.parallelize(my_y, 10)
coord = x.cartesian(x)
rdd = ( coord.map( lambda s: [s, mandelbrot(s[1]/500.0-2, s[0]/500.0-2)] ) ).persist()


#draw_image(rdd)
plt.hist( sum_values_for_partitions(rdd).collect() )
plt.savefig('P2b_hist.png')

