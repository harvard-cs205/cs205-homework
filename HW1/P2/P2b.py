from P2 import *
# from P2a import make_mandelbrot_image_and_hist

import findspark
findspark.init()
import numpy as np
import pyspark

sc = pyspark.SparkContext(appName='mandelbrot')


x_values = sc.parallelize(np.random.permutation(2000), 10)
y_values = sc.parallelize(np.random.permutation(2000), 10)
coords = x_values.cartesian(y_values)

def make_mandelbrot_image_and_hist(coords_rdd, hist_fname):
	value_for_coord = lambda v: v / 500. - 2
	make_output = lambda (x, y): ((x, y), mandelbrot(value_for_coord(y), value_for_coord(x)) )
	image = coords_rdd.map(make_output)

	work_data_rdd = sum_values_for_partitions(image)
	work_data = work_data_rdd.take(work_data_rdd.count())
	plt.hist(work_data)
	plt.title('Compute by Partition')
	plt.xlabel('Partition')
	plt.ylabel('Compute')
	plt.savefig(hist_fname)
	plt.clf()

	draw_image(image)

make_mandelbrot_image_and_hist(coords, 'P2b_hist.png')
