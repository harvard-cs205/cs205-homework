from P2 import *

# Your code here
from pyspark import SparkContext
import matplotlib.pyplot as plt
import numpy as np

sc = SparkContext("local[4]", "HW1-2b Default Partioning")
sc.setLogLevel("ERROR")

#set up 2000x2000 array and transform array into [-2,2]x[-2,2] grid... 
rdd = sc.parallelize(xrange(0,2001),10)
grid = rdd.cartesian(rdd).repartition(100) #Creates ordered pairs, redistribute randomly...

#pass grid values to mandelbrot.py and append output of mandelbrot.py to tuple --> (i,j,iter)
grid_Computed = grid.map(lambda x: (x,mandelbrot((x[1]/500.0)-2,(x[0]/500.0)-2 )))

#Run through visualization and metric measurement functions.
#draw_image(grid_Computed)

summedITER = sum_values_for_partitions(grid_Computed)
plt.hist(summedITER.collect(),20)
plt.xlabel("Iterations per Partition")
plt.ylabel("Partitions")
plt.title("Mandelbrot Set Calculation with Random Partitioning")
plt.savefig('P2b_hist.png')
plt.show()