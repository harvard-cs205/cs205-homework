from P2 import *

# Create 2000 x 2000 tuples in an array representing positions of all pixels
board = []
for i in xrange(2000):
	for j in xrange(2000):
		board.append((i,j))

# Randomize the positions of pixels in the list
sc = pyspark.SparkContext()
rdd = sc.parallelize(board)
rdd = rdd.repartition(100)
rddMandelbrot = rdd.map(lambda (i,j): ((i,j), mandelbrot(j/500.0 - 2, i/500.0 - 2)))

# Draw the image
draw_image(rddMandelbrot)

# Produce histogram of compute effort on each partition
computeEffort = sum_values_for_partitions(rddMandelbrot)
plt.hist(computeEffort.collect())
plt.xlabel('Compute effort')
plt.ylabel('Partitions')
plt.title('Histogram for randomized partitioning')
plt.savefig('Randomized-partitioning')