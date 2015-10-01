# Default partitioning: 100 partitions!

import matplotlib.pyplot as plt

import P2
import seaborn as sns
sns.set_context('poster', font_scale=1.25)
import findspark as fs
fs.init()
import pyspark as ps
import multiprocessing as mp
import numpy as np

######## Same as part A, just setting up ################

# Setup cluster, number of threads = 2x cores
config = ps.SparkConf()
config = config.setAppName('P2a')

sc = ps.SparkContext(conf=config)

# Do the computation

num_pixels = 2000
rows = sc.range(num_pixels, numSlices=10)
cols = sc.range(num_pixels, numSlices=10)

indices = rows.cartesian(cols)

def mandelbrot_wrapper(row, col):
    x = col/(num_pixels/4.) - 2.
    y = row/(num_pixels/4.) - 2.

    return ((row, col), P2.mandelbrot(x, y))

########### Different from part A: load balancing! ########
new_indices = indices.repartition(100)

mandelbrot_load_balanced = new_indices.map(lambda a: mandelbrot_wrapper(*a))

summed_rdd = P2.sum_values_for_partitions(mandelbrot_load_balanced)
summed_result = summed_rdd.collect()

# Now collect the data & plot
plt.hist(summed_result, bins=np.logspace(3, 8, 20))
sns.rugplot(summed_result, color='red')
plt.gca().set_xscale('log')
plt.xlabel('Total Number of Iterations on Partition')
plt.ylabel('Partition Count')
plt.title('Number of Iterations on each Partition')

plt.savefig('P2b_alternative_hist.png', dpi=200, bbox_inches='tight')
