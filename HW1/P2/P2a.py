from P2 import *

# Creating 2000 x 2000 tuples in an array representing positions of all pixels
sc = pyspark.SparkContext()
board = range(2000)

# Divide mandelbrot computation into 100 partitions
rdd = sc.parallelize(board, 10)
rdd = rdd.cartesian(rdd)
rddMandelbrot = rdd.map(lambda (i,j): ((i,j), mandelbrot(j/500.0 - 2, i/500.0 - 2)))

# Draw image
draw_image(rddMandelbrot)

# Produce histogram of compute effort on each partition
computeEffort = sum_values_for_partitions(rddMandelbrot)
plt.hist(computeEffort.collect(), bins=10)
plt.xlabel('Compute effort')
plt.ylabel('Partitions')
plt.title('Histogram for sequential partitioning')
plt.savefig('Sequential-partitioning')