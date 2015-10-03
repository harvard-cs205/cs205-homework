from P2 import *

# Your code here
import pyspark
from pyspark import SparkContext, SparkConf
sc = SparkContext()

x = sc.parallelize(xrange(1,2001,1), 10)
coord = x.cartesian(x)
rdd = ( coord.map( lambda s: [s, mandelbrot(s[1]/500.0-2, s[0]/500.0-2)] ) ).persist()


draw_image(rdd)
plt.hist( sum_values_for_partitions(rdd).collect() )
plt.savefig('P2a_hist.png')
