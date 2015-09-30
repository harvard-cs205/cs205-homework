from P2 import *
import findspark
findspark.init()
import os
import pyspark
sc = pyspark.SparkContext()

# define axis rdd with 10 partitions to prepare for a square matrix with 100 patitions
axisrdd = sc.parallelize(range(2000),10)
pixelrdd = axisrdd.cartesian(axisrdd)

# apply the mandelbrot function and draw image
drawrdd = pixelrdd.map(lambda pixel: [pixel, mandelbrot(pixel[1]/500.0-2, pixel[0]/500.0-2)])
draw_image(drawrdd)

# plot effort with histogram
plt.hist(sum_values_for_partitions(drawrdd).collect())
plt.savefig('P2a_hist.png')