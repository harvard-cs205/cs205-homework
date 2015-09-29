from P2 import *
import random

parts = range(100)
def choose(*args):
    return random.choice(parts)

import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext()

a = np.arange(2000)
b = np.arange(2000)
c = sc.parallelize(a, 10).cartesian(sc.parallelize(b, 10)).partitionBy(100, choose)
d = c.map(lambda x: (x, mandelbrot(x[0] / 500.  - 2, x[1] / 500. - 2)))
draw_image(d)

e = sum_values_for_partitions(d)
f = e.collect()
plt.hist(f)
plt.show()
