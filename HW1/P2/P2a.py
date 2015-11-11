from __future__ import division
from P2 import *
import pyspark
import time


sc = pyspark.SparkContext(appName = "Spark1")

# Use default partitioning below
yaxis = sc.parallelize([ii for ii in range(0,2000)],10)#.map(lambda x: (x,x))
xaxis = sc.parallelize([ii for ii in range(0,2000)],10)#.map(lambda x: (x,x))
#xaxis = xaxis.partitionBy(10,lambda x: int(np.floor(x/200)))
#xaxis = xaxis.map(lambda x: x[0])

# form image using cartesian command for cartesian product
image = xaxis.cartesian(yaxis).cache()

newImage = image.map(lambda x: ((x[1],x[0]),mandelbrot((x[0]/500)-2,(x[1]/500)-2))).cache()

# draw image and compute worker time of 100 workers below
#draw_image(newImage)

startTime = time.time()
workTime = sum_values_for_partitions(newImage).collect()
endTime = time.time()

plt.hist(workTime)
plt.title('Default Worker Time Distribution')
plt.xlabel('Number of Iterations')
plt.ylabel('Count')
plt.show()

print endTime - startTime




"""

aa = sc.parallelize([ii for ii in range(0,10)],10).map(lambda x: (x,x))
bb = sc.parallelize([ii for ii in range(0,10)],10).map(lambda x: (x,x))
cc = aa.join(bb)
print cc.glom().collect()


aa = sc.parallelize([ii for ii in range(0,10)]).map(lambda x: (x,x)).partitionBy(10)
bb = sc.parallelize([ii for ii in range(0,10)]).map(lambda x: (x,x)).partitionBy(10)
cc = aa.join(bb)
print len(cc.glom().collect())

print yaxis.glom().collect()
zaxis = sc.parallelize([ii for ii in range(0,2000)])
zaxis = zaxis.map(lambda x: (x,x))
zaxis = zaxis.partitionBy(10,lambda x: int(np.floor(x/200)))
zaxis = zaxis.map(lambda x: x[1])
print zaxis.glom().collect()

"""

