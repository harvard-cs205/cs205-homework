from pyspark import SparkContext
from P2 import *
sc = SparkContext()
x=sc.parallelize(xrange(1,2001,1),10)
final=x.cartesian(x)
rdd=(final.map(lambda r: [r,mandelbrot(r[1]/500.0-2,r[0]/500.0-2)])).persist()

draw_image(rdd)
val=sum_values_for_partitions(rdd)
plt.hist(val.collect())
plt.savefig('P2a_hist.png')
