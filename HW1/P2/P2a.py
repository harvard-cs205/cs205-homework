from P2 import *

# Your code here
import findspark
findspark.init()
import pyspark

sc = pyspark.SparkContext(appName="Spark2a")

# make pyspark shut up
sc.setLogLevel('WARN')

imagei = sc.parallelize([x for x in range(2000)], 10)
image = imagei.cartesian(imagei)

mandelbrots = image.map(lambda (i,j): ((i, j), mandelbrot(j/500.0 - 2, i/500.0 - 2)))

iterations_by_partition = sum_values_for_partitions(mandelbrots).collect()
plt.hist(iterations_by_partition, 50)
plt.xlabel("iterations")
plt.ylabel("number of partitions")
plt.show()


