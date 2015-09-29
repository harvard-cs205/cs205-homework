from P2 import *

# Your code here
import findspark
findspark.init('/Users/georgelok/spark')

import pyspark
sc = pyspark.SparkContext(appName="myAppName")

def cartesianToPolarAngle(x,y,factor) :
  ratio = x / (y * 1.0)
  angle = np.arctan(ratio)
  if(x < 0) :
    angle += np.pi
  return int(angle / (2 * np.pi) * 100)%100

# Set 100 partitions
coordinates = sc.parallelize([(x,y) for x in xrange(2000) for y in xrange(2000)])
coordinates = coordinates.partitionBy(100)
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
plt.title('Compute Effort with polar partitioning')
plt.savefig('P2b_hist.png')
plt.show()
