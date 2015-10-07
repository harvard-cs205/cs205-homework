"""
    Written by Jaemin Cheun
    Harvard CS205
    Assignment 1
    October 6, 2015
    """

from P2 import *
import numpy as np
import findspark
findspark.init()

from pyspark import SparkContext

# initiaize Spark
sc = SparkContext("local", appName="P2a")

# Image of 2000 x 2000 pixels
dim = 2000

def pixel(x): 
	return (x/500.0) - 2

# Partition by 10 because when using cartesian, it will be partitioned into 10 x 10 = 100 partitions total
rdd = sc.parallelize(xrange(dim), 10)

# Take the cartesion coordinate, but repartition here 
coord = rdd.cartesian(rdd).repartition(100)

# Calculate the pixel
mandel_RDD = coord.map(lambda (i,j): ((i,j), mandelbrot(pixel(i),pixel(j))))

# how much computation was done in each partition
sum_partition = sum_values_for_partitions(mandel_RDD).collect()

# First Plot: Iterations per Partition vs Number of Partitions
plt.figure()
plt.hist(sum_partition)
plt.title("Computation of Mandelbrot Set using the random partitioning strategy")
plt.xlabel("Iterations per Partition")
plt.ylabel("Number of Partitions")
plt.savefig('P2b_hist.png')

# Second Plot: Partition Index vs Iterations per Partition
partition = np.arange(len(sum_partition))
plt.figure()
plt.bar(partition,sum_partition)
plt.xlabel("Partition Index")
plt.ylabel("Iterations per Partition")
plt.title("Computation of Mandelbrot Set using the random partitioning strategy")
plt.savefig('P2b_bar.png')
