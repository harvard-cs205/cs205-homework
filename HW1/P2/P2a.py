from P2 import *

# # Your code here

# import sys
# sys.path.append('/Users/devvret/Desktop/Senior Year/CS205/findspark-master')

# import findspark

# findspark.init()

import pyspark
from pyspark import SparkContext


sc = SparkContext()

#create the X and Y coordinates

X = sc.parallelize(xrange(2000), 10)
Y = sc.parallelize(xrange(2000), 10)

#create the cartesian grid
cart = X.cartesian(Y)

#create the mapping RDD
mapping = cart.map(lambda (i, j): ((i,j), mandelbrot((i/500.0)-2, (j/500.0) - 2)))

#draw the mapping RDD

draw_image(mapping)

#compute the work

compute = sum_values_for_partitions(mapping)

#plot the histogram

plt.hist(compute.collect())
plt.title('Compute on Default Partions')
#plt.show()
plt.savefig('P2a_hist.png')



