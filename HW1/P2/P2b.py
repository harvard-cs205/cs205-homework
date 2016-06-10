from P2 import *
import findspark
findspark.init()
import pyspark
import random
from random import randint
sc = pyspark.SparkContext(appName="Spark1")
x_ = range(2000)
random.shuffle(x_)
x = sc.parallelize(x_,10)
rdd = x.cartesian(x)
#y = sc.parallelize(xrange(2000),10)
#rdd = x.cartesian(x).repartition(100)
#rdd = x.cartesion(x).partitionBy(lambda x:randint(1,101))
ans = rdd.map(lambda x: [x,mandelbrot((x[0]/500.0-2),(x[1]/500.0-2))])
draw_image(ans)

ans1 = sum_values_for_partitions(ans)

plt.hist(ans1.collect())
plt.savefig('P2b_hist.png')