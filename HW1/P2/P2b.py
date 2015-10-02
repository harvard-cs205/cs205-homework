from P2 import *

# Creating 2000 x 2000 tuples in an array representing positions of all pixels
board = []
for i in xrange(2000):
	for j in xrange(2000):
		board.append((i,j))

# Randomize the positions of pixels in the list
sc = pyspark.SparkContext()
rdd = sc.parallelize(board)
rdd1 = rdd.repartition(100)
rdd2 = rdd1.map(lambda (i,j): ((i,j), mandelbrot(j/500.0 - 2, i/500.0 - 2)))

# Draw the image
draw_image(rdd2)

# Produce histogram of compute effort on each partition
computeEffort = sum_values_for_partitions(rdd2)
plt.hist(computeEffort.collect())
plt.savefig('Randomized partitioning')

# # Divide mandelbrot computation into 100 partitions
# rdd = sc.parallelize(board)
# rdd1 = rdd.zipWithIndex()
# # Want to change index to the index modulo 100 (# of partitions)
# rdd2 = rdd1.map(lambda (pos, idx): (pos, idx % 100))
# rdd2.sort()