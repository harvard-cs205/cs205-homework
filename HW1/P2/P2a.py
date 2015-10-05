# Author: George Lok
# P2a.py

from P2 import *

# Your code here
import findspark
findspark.init('/Users/georgelok/spark')

import pyspark
sc = pyspark.SparkContext(appName="myAppName")

# Set 100 partitions
coordinates = sc.parallelize([(x,y) for x in xrange(2000) for y in xrange(2000)], 100)
results = coordinates.map(lambda (x,y) : ((x,y), mandelbrot((x/500.0) - 2., (y/500.0)-2.)))

# Confirm result is correct
draw_image(results)

computeData = sum_values_for_partitions(results)

data = computeData.collect()
effort = np.array([d for d in data])

plt.hist(range(len(effort)), 100, weights=effort)
plt.yscale('log')
plt.xlabel('Partition ID')
plt.ylabel('Log of Number of Iterations')
plt.title('Compute Effort with default partitioning')
plt.savefig('P2a_hist.png')
plt.show()
