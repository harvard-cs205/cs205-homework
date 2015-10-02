from P2 import *

import numpy as np

import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext(appName="P2")

pixels = np.zeros((2000, 2000))
print pixels

parallel_pix = sc.parallelize(pixels, 100)

#print parallel_pix.take(10)

computed = parallel_pix.map(lambda a: mandelbrot(a[0],a[1]))

print len(computed.collect())

#draw_image(computed)