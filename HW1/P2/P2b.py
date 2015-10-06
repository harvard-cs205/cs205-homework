from P2 import *
import findspark
findspark.init()

import pyspark

sc = pyspark.SparkContext(appName="P2")

#computes the Mandelbrot pixel at the point (i,j)
#where x = (j/500) - 2 and y = (i/500) - 2
def mandelbrot_ij(i,j):	
	return mandelbrot((j/500.0)-2, (i/500.0)-2)

#create an image of 2000 x 2000 pixels
image_width = 2000
image_height = 2000	

#create a list of x and y pixel values, in 10 partitions each
#these lists are now randomized, so the contents of each partition are also
#randomized rather than ascending
width_values = sc.parallelize(np.random.permutation(xrange(image_width)), 10)
height_values = sc.parallelize(np.random.permutation(xrange(image_height)), 10)

#calculates all pixel values, dividing the computation into 100 partitions
pixel_values = width_values.cartesian(height_values)

#compute the Mandelbrot, dividing the computation into 100 partitions
mandelbrot_results = pixel_values.map(lambda (x,y): ((x,y), mandelbrot_ij(x, y)))
draw_image(mandelbrot_results)

#compute "effort" on each partition
timing_results = sum_values_for_partitions(mandelbrot_results)

plt.hist(timing_results.collect())
plt.show()
plt.savefig("P2b_hist.png")

print "Done"
