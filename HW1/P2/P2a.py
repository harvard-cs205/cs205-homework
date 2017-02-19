from P2 import *
import numpy as np
from pyspark import SparkContext
import matplotlib.pyplot as plt

sc = SparkContext('local', "Mandelbrot")

#initializa the coordinate points as a list
coor = []
for i in range(0,2000):
	coor.append(i)


#parallelize it in spark with 10 partitions. cartesian product will make it 100
img = sc.parallelize(coor, 10)
img2 = img.cartesian(img)

#calculate the mandel image
mandel_image = img2.map(lambda (i,j): ((i,j), mandelbrot( ((j/500.0)-2), ((i/500.0)-2) ) )) 

#sum of values in partitions
sum_data = sum_values_for_partitions(mandel_image)
plt.hist(sum_data.collect())
plt.show()

#uncomment this to draw the image
#draw_image(mandel_image)


