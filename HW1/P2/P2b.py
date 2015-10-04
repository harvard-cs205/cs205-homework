from pyspark import SparkContext
import random
from P2 import *

sc = SparkContext()
x=range(2000)
random.shuffle(x)
y=range(2000)
random.shuffle(y)
xx=sc.parallelize(x,10)
yy=sc.parallelize(x,10)
final=xx.cartesian(yy)
rdd=final.map(lambda r: [r,mandelbrot(r[1]/500.0-2,r[0]/500.0-2)])

val=sum_values_for_partitions(rdd)
plt.hist(val.collect())
plt.savefig('P2b_hist.png')
