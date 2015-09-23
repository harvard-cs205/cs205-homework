from __future__ import division
from P2 import *
import pyspark
import time

sc = pyspark.SparkContext(appName = "Spark1")

yaxis = sc.parallelize([(ii/500)-2 for ii in range(0,2000)],10)
xaxis = sc.parallelize([(ii/500)-2 for ii in range(0,2000)],10)

#taxis = xaxis.mapPartitions(lambda x: [len(np.array(list(x)))]).collect()
#print taxis

image = xaxis.cartesian(yaxis)

newImage = image.map(lambda x: [[x[0],x[1]],mandelbrot(x[0],x[1])]).cache()
#print np.shape(newImage)

draw_image(newImage)
workTime = sum_values_for_partitions(newImage)

print workTime.collect()




