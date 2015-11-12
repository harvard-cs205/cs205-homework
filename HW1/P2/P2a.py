from P2 import *

# Initialize Spark
from pyspark import SparkContext
sc = SparkContext()
sc.setLogLevel("ERROR")

# Define key parameters
coordWidth = 2000
coordHeight = 2000
noPartitions = 100

# Create RDD with i, j coordinates & set number of partitions
pixCoordIJ = sc.parallelize([(x, y) for x in range(coordWidth) for y in range(coordHeight)], noPartitions)

# Apply the Mandelbrot function
pixMandel = pixCoordIJ.map(lambda (i, j): ((i, j), mandelbrot(j/500.0 - 2, i/500.0 - 2)))

# Draw resulting image (using helper function)
draw_image(pixMandel)

# Calculate effort per partition
partitionWork = sum_values_for_partitions(pixMandel).collect()

# Plot effort per partition
plt.figure()
plt.hist(partitionWork)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlabel('Computation per partition')
plt.ylim(0,70)
plt.ylabel('Frequency')
plt.title('P2A: Per-partition work (default partitioning strategy)')
plt.savefig('P2a_hist.png')
plt.show()