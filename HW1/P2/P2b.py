from P2 import *

# Your code here

import pyspark
from pyspark import SparkContext
import random


sc = SparkContext()

#create the x and y ranges
X = range(2000)
Y = range(2000)

#randomly shuffle the X and Y to 
#more evenly distribute the load

random.shuffle(X)
random.shuffle(Y)

#same steps as last time. 

x = sc.parallelize(X, 10)
y = sc.parallelize(Y, 10)

cart = x.cartesian(y)

mapping = cart.map(lambda (i, j): ((i,j), mandelbrot((i/500.0)-2, (j/500.0) - 2)))

#draw_image(mapping)

compute = sum_values_for_partitions(mapping)

plt.hist(compute.collect())
plt.title('Compute on Randomized Partions')
#plt.show()
plt.savefig('P2b_hist.png')
