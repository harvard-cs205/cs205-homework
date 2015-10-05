######################################################
### Problem 2 - Computing the Mandelbrot Set [10%] ###
### P2a.py									       ###
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

	
	# Create RDD (ensure 100 partitions with numslices=10*10) and use Cartesian to Create Pixels
	x = sc.parallelize(xrange(2000), 10)
	y = sc.parallelize(xrange(2000), 10)
	all_pixels = x.cartesian(y)

	# Call Mandlebrot Function and Draw Image
	mandelbrot_rdd = all_pixels.map(lambda p: (p, mandelbrot((p[0]/500.) - 2, (p[1]/500.) - 2)))
	draw_image(mandelbrot_rdd)

	# Find the number of all paritions
	mandel_parts = sum_values_for_partitions(mandelbrot_rdd)
	mandel_hist_rdd = mandel_parts.collect()

	# Plot and Save
	plt.figure(figsize=(8,5))
	plt.hist(mandel_hist_rdd)
	plt.title("Default RDD Partition Histogram")
	plt.xlabel("Computitions")
	plt.ylabel("Number of Paritions")
	plt.savefig('P2a_hist.png')
	plt.show()