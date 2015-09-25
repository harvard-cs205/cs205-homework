from P2 import *
import findspark
findspark.init()
import pyspark
from random import randint
sc = pyspark.SparkContext(appName="Spark1")
x = sc.parallelize(xrange(2000),10)
y = sc.parallelize(xrange(2000),10)
rdd = x.cartesian(x).repartition(100)

ans = rdd.map(lambda x: [x,mandelbrot((x[0]/500.0-2),(x[1]/500.0-2))])
draw_image(ans)

ans1 = sum_values_for_partitions(ans)

plt.hist(ans1.collect())
plt.savefig('P2b_hist.png')