from P2 import *

# Your code here
import pyspark
import numpy as np

sc = pyspark.SparkContext(appName='myAppName')

i = sc.parallelize(xrange(1,2001), 10)
j = sc.parallelize(xrange(1,2001), 10)

coordinates = i.cartesian(j)

intensity = coordinates.map(lambda x: [x, mandelbrot(x[1]/500.0-2, x[0]/500.0-2)])

draw_image(intensity)