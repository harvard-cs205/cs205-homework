from P2 import *
from pyspark import SparkContext, SparkConf
import matplotlib.pyplot as plt

#Setup
conf = SparkConf().setAppName("Mandelbrot").setMaster("local")
sc = SparkContext(conf=conf)

#Create x and y axes each with 2000 points divided into 10 partitions according to the default hash function, then convert to a grid
#-1 is a dummy value required to call partitionBy()
xs = sc.range(2000).map(lambda x: (x, -1)).partitionBy(10)
ys = xs.map(lambda x: x)
plane = xs.cartesian(ys)

#Calculate Mandelbrot for each point on the grid
mandelbrot2d = plane.map(lambda (K, V): ((K[0], V[0]), mandelbrot((K[0]/500.0) - 2, (V[0]/500.0) - 2)))
assert mandelbrot2d.getNumPartitions() == 100

#Sum repetitions for each partition and plot
summed = sum_values_for_partitions(mandelbrot2d)
plt.hist(summed.collect())
plt.savefig("P2b.png")
draw_image(mandelbrot2d)
