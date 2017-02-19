from P2 import *
import numpy as np
from pyspark import SparkContext
import matplotlib.pyplot as plt

from random import shuffle

sc = SparkContext('local', "Mandelbrot")



#initializa all the coordinate points as a list
#As I will be shuffling the coordinate points to balance the work, i won't be using cartesian product (like in P2a)
coor = []
for i in range(0,2000):
	for j in range(0,2000):
		coor.append((i,j))


#randomized the positions of coordinates in coor
shuffle(coor)



#parallelize it in spark with 100 partitions
img = sc.parallelize(coor, 100)

#calculate the mandel image
mandel_image = img.map(lambda (i,j): ((i,j), mandelbrot(((j/500.0)-2), ((i/500.0)-2)))) 

#sum of values in partitions
sum_data = sum_values_for_partitions(mandel_image)
plt.hist(sum_data.collect())
plt.show()

#uncomment this to draw the image
#draw_image(mandel_image)


