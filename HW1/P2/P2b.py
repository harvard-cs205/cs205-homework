from P2 import *
from pyspark import SparkContext
import random

sc = SparkContext("local", "Simple App")

array_1 = [i for i in range(2000)]
array_2 = [i for i in range(2000)]

new_array = []
for i in array_1:
	for j in array_2:
		new_array.append((j,i))

intermediate = sc.parallelize(new_array)
intermediate = intermediate.partitionBy(100, lambda x : random.randint(0,99)).map(lambda x: ((x[0], x[1]), mandelbrot(x[0]/500.0 - 2, x[1]/500.0 - 2)))
sum_values = sum_values_for_partitions(intermediate)
print "SUM VALUES IS"
print sum_values.take(10)
#intermediate = map(lambda x: ((x[0], x[1]), mandelbrot(x[0]/500.0 - 2, x[1]/500.0 - 2)), new_array)
#print intermediate
plt.hist(sum_values.collect())
plt.ylabel('Number of Partitions')
plt.xlabel('Total iterations in Partition')
plt.show()
draw_image(intermediate)