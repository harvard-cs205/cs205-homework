from P2 import *
from pyspark import SparkContext
import random
sc =SparkContext()

# Your code here

# Function used to map iterations
def getXY(K):
	x = (K[1]/500.) - 2
	y = (K[0]/500.) - 2

	return K, mandelbrot(x,y)

# Gernate coordinates
rdd = sc.parallelize(range(0,2000),10)

# Performs a shuffle to randomize the allocation of coordinates to partitions
coordinates = rdd.cartesian(rdd).partitionBy(100, lambda coord: random.randint(0,99))

rddout = coordinates.map(getXY)

# Draw image (Not useful here)
draw_image(rddout)

# Evaluate the performance of each partition
Performance = sum_values_for_partitions(rddout)
res = Performance.collect()
plt.hist(res, bins = 10, color = "steelblue")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel("Iterations")
plt.ylabel("Freq")
plt.title("Partition 2b")
plt.savefig("P2b_hist.png")
plt.show()

