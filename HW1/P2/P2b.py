from __future__ import division
from P2 import *
import pyspark
import time

sc = pyspark.SparkContext(appName = "Spark1")

baseList = np.array([ii for ii in range(0,2000)])
np.random.seed(100)
rList = np.random.uniform(0,1,2000)
rPerm = np.array(sorted(range(len(rList)), key = lambda k:rList[k]))
repNum = np.array([ii for ii in range(0,10)]*200)
rDictX = repNum[rPerm]
np.random.seed(200)
rList = np.random.uniform(0,1,2000)
rPerm = np.array(sorted(range(len(rList)), key = lambda k:rList[k]))
repNum = np.array([ii for ii in range(0,10)]*200)
rDictY = repNum[rPerm]

yaxis = sc.parallelize(baseList).map(lambda x: (x,x))
yaxis = yaxis.partitionBy(10,lambda x: rDictY[x])
yaxis = yaxis.map(lambda x: x[0])

xaxis = sc.parallelize(baseList).map(lambda x: (x,x))
xaxis = xaxis.partitionBy(10,lambda x: rDictX[x])
xaxis = xaxis.map(lambda x: x[0])

image = xaxis.cartesian(yaxis).cache()

newImage = image.map(lambda x: [[x[1],x[0]],mandelbrot((x[0]/500)-2,(x[1]/500)-2)]).cache()

#draw_image(newImage)

workTime = sum_values_for_partitions(newImage).collect()

plt.hist(workTime)
plt.title('Custom Worker Time Distribution')
plt.show()


