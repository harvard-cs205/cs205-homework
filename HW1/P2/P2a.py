from P2 import *
import matplotlib.pyplot as plt
from pyspark import SparkContext

# constants
GRID_DIM = 2000
PART_DIM = 10
PLOT_BINS = 20

sc = SparkContext("local", "Mandelbrot default")

# create RDD with default partitioning
dim_1 = sc.parallelize(xrange(GRID_DIM), PART_DIM)
grid = dim_1.cartesian(dim_1)

# calculate mandlebrot set
mand_in = grid.map(lambda (i, j): ((i, j), (j / 500.0 - 2, i / 500.0 -2)))
mand_out = mand_in.mapValues(lambda (x, y): mandelbrot(x, y))
draw_image(mand_out)

# calculate amount of work for each partition, plot as histogram
tot_iters = sum_values_for_partitions(mand_out)
plt.hist(tot_iters.collect(), PLOT_BINS)

# draw plot
plt.xlabel("Iterations")
plt.ylabel("Partition")
plt.title("Mandlebrot Set Calculation with Default Partitioning")
plt.show()
