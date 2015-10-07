from __future__ import division
from P2 import *
import pyspark
import time


sc = pyspark.SparkContext(appName = "Spark1")

baseList = np.array([ii for ii in range(0,2000)])
np.random.seed(100)
rList = np.random.uniform(0,1,2000)
rPerm = np.array(sorted(range(len(rList)), key = lambda k:rList[k])) # get random permutation of 1,...,2000
repNum = np.array([ii for ii in range(0,10)]*200) # get 200 copies of each hash key (1,...,10)
rDictX = repNum[rPerm] # form hash map using random permutation of hash keys, same for y values below
np.random.seed(200)
rList = np.random.uniform(0,1,2000)
rPerm = np.array(sorted(range(len(rList)), key = lambda k:rList[k]))
repNum = np.array([ii for ii in range(0,10)]*200)
rDictY = repNum[rPerm]

yaxis = sc.parallelize(baseList).map(lambda x: (x,x)) 
yaxis = yaxis.partitionBy(10,lambda x: rDictY[x]) # paritition using random hashing
yaxis = yaxis.map(lambda x: x[0])

xaxis = sc.parallelize(baseList).map(lambda x: (x,x))
xaxis = xaxis.partitionBy(10,lambda x: rDictX[x]) # partition using random hashing
xaxis = xaxis.map(lambda x: x[0])

image = xaxis.cartesian(yaxis).cache()

newImage = image.map(lambda x: [[x[1],x[0]],mandelbrot((x[0]/500)-2,(x[1]/500)-2)]).cache()

#draw_image(newImage)
startTime = time.time()
workTime = sum_values_for_partitions(newImage).collect()
endTime = time.time()

plt.hist(workTime)
plt.title('Custom Worker Time Distribution')
plt.show()

print endTime - startTime
