from P2 import *
from pyspark import SparkContext, SparkConf
import matplotlib.pyplot as plt

#Setup
conf = SparkConf().setAppName("Mandelbrot").setMaster("local[*]")
sc = SparkContext(conf=conf)

#Create x and y axes each with 2000 points divided into 10 partitions, then convert to a grid
xs = sc.range(2000, numSlices=10)
ys = xs.map(lambda x: x)
plane = xs.cartesian(ys)

#Calculate Mandelbrot for each point on the grid
mandelbrot2d = plane.map(lambda (x, y): ((x, y), mandelbrot((x/500.0) - 2, (y/500.0) - 2)))
assert mandelbrot2d.getNumPartitions() == 100

#Sum repetitions for each partition and plot
summed = sum_values_for_partitions(mandelbrot2d)
plt.hist(summed.collect())
plt.savefig("P2a_hist.png")
draw_image(mandelbrot2d)
