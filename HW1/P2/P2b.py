from P2 import *
import matplotlib.pyplot as plt
from pyspark import SparkContext
import random

# constants
GRID_DIM = 2000
PART_DIM = 10
PLOT_BINS = 20

sc = SparkContext("local", "Mandelbrot load balanced")

# randomize coordinates first, then partition as in default
rand_coords = range(GRID_DIM)
random.shuffle(rand_coords)
dim_1 = sc.parallelize(rand_coords, PART_DIM)
grid = dim_1.cartesian(dim_1)

# calculate mandlebrot set
mand_in = grid.map(lambda (i, j): ((i, j), (j / 500.0 - 2, i / 500.0 -2)))
mand_out = mand_in.mapValues(lambda (x, y): mandelbrot(x, y))

# calculate amount of work for each partition, plot as histogram
tot_iters = sum_values_for_partitions(mand_out)
plt.hist(tot_iters.collect(), PLOT_BINS)

# draw plot
plt.xlabel("Iterations")
plt.ylabel("Partition")
plt.title("Mandlebrot Set Calculation with Load Balancing (Random)")
plt.show()
