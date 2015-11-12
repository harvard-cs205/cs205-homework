import findspark 
findspark.init()

from P2 import *
from pyspark import SparkContext

sc = SparkContext()

#create lists of the range for rows and cols  
ilist = range(2000)
jlist = range(2000)

#turn the lists into 1 rdd, K: (i,j)
rddi = sc.parallelize(ilist,10)
rddj = sc.parallelize(jlist,10)
rddK = rddi.cartesian(rddj)

#create values V for map
mandelrdd = rddK.map(lambda (x,y): ((x, y), mandelbrot(y/500.0-2, x/500.0-2)))

#create the image
draw_image(mandelrdd)

#plot this histogram of the number of partitions made
sum_parts = sum_values_for_partitions(mandelrdd).collect()
plt.hist(sum_parts)
plt.title("P2a Histogram - Sequential Partitions")
plt.savefig("P2a_hist.png")
plt.show()