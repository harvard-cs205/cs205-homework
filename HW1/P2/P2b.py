from P2 import *

# Your code here
import findspark
findspark.init('/home/raphael/spark')

import pyspark
sc = pyspark.SparkContext(appName="myAppName")

T=sc.parallelize(np.array([k for k in range(2000)]))
U=sc.parallelize(np.array([k for k in range(2000)]))
mesh=T.cartesian(U)
#give key in a smart way so that partitionBy(100) take a better partition, it's a way to impose a new hash
meshKV = mesh.map(lambda point : (point[1]%100,point))
mesh2 = meshKV.partitionBy(100)
f=lambda x: x/500.-2
mandel = mesh2.map(lambda (K,V): (V,mandelbrot(f(V[0]),f(V[1]))))
#action against lazyness
print(mesh2.count(), mandel.count())
hist = sum_values_for_partitions(mandel)
load = hist.take(100)

plt.hist(load,bins=10);
plt.ylabel("number of workers")
plt.xlabel("number of operations")
plt.title("histogram of the load balancing with optimization")
plt.savefig("/home/raphael/cs205-homework/HW1/P2/P2b_hist.png")
