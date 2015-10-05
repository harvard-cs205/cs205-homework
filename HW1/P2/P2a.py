from P2 import *

# Your code here
import findspark
findspark.init('/home/raphael/spark')

import pyspark
sc = pyspark.SparkContext(appName="myAppName")

test=sc.parallelize(np.array([1, 2, 3, 4]))
T=sc.parallelize(np.array([k for k in range(2000)]),10)
U=sc.parallelize(np.array([k for k in range(2000)]),10)
mesh=T.cartesian(U) #has 100 partitions
f=lambda x: x/500.-2
mandel = mesh.map(lambda point : (point, mandelbrot(f(point[0]),f(point[1]))))

hist = sum_values_for_partitions(mandel)
load = hist.take(100)

plt.hist(load,bins=10);
plt.ylabel("number of workers")
plt.xlabel("number of operation")
plt.title("histogram of the load balancing without optimization")
plt.savefig("/home/raphael/cs205-homework/HW1/P2/P2a_hist.png")
plt.show()
