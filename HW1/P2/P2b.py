from P2 import *

# Your code here
from pyspark import SparkContext

# initialize spark
sc = SparkContext()

# create rdd to save image pixels
n = 2000
rows = sc.parallelize(range(n), 10)
image = rows.cartesian(rows)

# assign partitions randomly
image = image.partitionBy(100, lambda part: np.random.randint(0,100))
print 'Number of partitions:', image.getNumPartitions()

# Mandelbrot computation
image = image.map(lambda (i, j): ((i,j), (j/500.-2, i/500.-2)))
image = image.mapValues(lambda v: mandelbrot(v[0], v[1]))
#draw_image(image)

# Plot histogram of iterations by partition
iters_by_partition = sum_values_for_partitions(image).collect()
plt.hist(iters_by_partition)
plt.title('Histogram of Iterations by Partition')
plt.ylabel('Frequency of Partitions')
plt.xlabel('Number of Iterations')
plt.savefig('P2b_hist.png')