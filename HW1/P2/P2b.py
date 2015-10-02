from P2 import *

# Your code here
import findspark
findspark.init()
import pyspark

sc = pyspark.SparkContext(appName="Spark2a")

imagei = sc.parallelize([x for x in range(2000)], 10)
image = imagei.cartesian(imagei)

mandelbrots = image.map(lambda (i,j): ((i, j), mandelbrot(j/500.0 - 2, i/500.0 - 2)))
mandelbrots_repartitioned = mandelbrots.partitionBy(100, lambda (i,j): int(np.random.uniform(0, 100)))

iterations_by_partition = sum_values_for_partitions(mandelbrots_repartitioned).collect()
plt.hist(iterations_by_partition, 50)
plt.xlabel("iterations")
plt.ylabel("number of partitions")
plt.show()


