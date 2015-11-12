######################################################
### Problem 2 - Computing the Mandelbrot Set [10%] ###
### P2b.py									       ###
### Patrick Day 								   ###
### CS 205 HW1                                     ###
### Oct 4th, 2015								   ###
######################################################

########################
### Import Functions ###
########################
import pyspark
import numpy as np
from P2 import * 

import os
sc = pyspark.SparkContext()
sc.setLogLevel('WARN')


########################
### Main Function    ###
########################
if __name__ == '__main__':

	# Random permutate the data
	rand_xy = np.random.permutation(xrange(2000))

	# Create RDD and use Cartesian to Create Pixels
	x = sc.parallelize(rand_xy, 10)
	y = sc.parallelize(rand_xy, 10)
	all_pixels = x.cartesian(y)

	# Call Mandlebrot Function
	mandelbrot_rdd = all_pixels.map(lambda p: (p, mandelbrot((p[0]/500.) - 2, (p[1]/500.) - 2)))
	
	# Repartition the Data and Draw Image
	mandel_repart = mandelbrot_rdd.repartition(100)
	draw_image(mandel_repart)

	# Find the number of all paritions
	sum_mandel_repart = sum_values_for_partitions(mandel_repart)
	mandel_hist_repart = sum_mandel_repart.collect()

	# Plot and Save
	plt.figure(figsize=(8,5))
	plt.hist(mandel_hist_repart)
	plt.title("RDD Partitioned by 100 Histogram")
	plt.xlabel("Computitions")
	plt.ylabel("Number of Paritions")
	plt.savefig('P2b_hist.png')
	plt.show()
