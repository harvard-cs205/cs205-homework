from P2 import *

# Your code here
import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
sc = SparkContext()

allxys = []

for x in range(0, 2000):
    for y in range(0, 2000):
        allxys.append((y, x))
        
rdd1 = sc.parallelize(allxys, 100).map(lambda xy: ((xy), mandelbrot((xy[1]/500.)-2,(xy[0]/500.)-2)))

draw_image(rdd1)
plt.hist(sum_values_for_partitions(rdd1).collect())