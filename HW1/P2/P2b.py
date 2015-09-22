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
config = config.setMaster('local[' + str(2*mp.cpu_count()) + ']')
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

# We create a mask of the most expensive computations. These are computations
# that take about 3 orders of magnitude longer to run than the fastest points!
expensive_mask = np.zeros((num_pixels, num_pixels))
e_top = 550
e_bottom = 1450
e_left = 300
e_right = 1250

expensive_mask[e_top:e_bottom, e_left:e_right] = 1
expensive_mask[1250:e_bottom, e_left:650]=0
expensive_mask[e_top:750, e_left:650]=0
expensive_mask[990:1010, 0:e_left]=1

# We broadcast the expensive mask to avoid communication overhead
# not sure if this is technically necessary
broadcast_expensive = sc.broadcast(expensive_mask)

# Determine if the index pair is expensive
indices_vs_expensive = indices.map(lambda a: (a, broadcast_expensive.value[a[0], a[1]]))

mandelbrot_rdd = indices.map(lambda a: mandelbrot_wrapper(*a))

# Now collect the data & plot
mandelbrot_result = mandelbrot_rdd.collect()

plt.grid(False)
# I slightly redefined the draw image function as the original
# implementation annoyed me...I did not want to collect in a draw function!
P2.draw_image(data=mandelbrot_result)

plt.savefig('P2a_mandelbrot.png', dpi=200, bbox_inches='tight')

plt.clf()

# Now create the histogram...I recognize that mandelbrot is computed twice
# but it is for my sanity
summed_rdd = P2.sum_values_for_partitions(mandelbrot_rdd)
summed_result = summed_rdd.collect()

plt.hist(summed_result, bins=np.logspace(3, 8, 20))
sns.rugplot(summed_result, color='red')
plt.gca().set_xscale('log')
plt.xlabel('Total Number of Iterations on Partition')
plt.ylabel('Partition Count')
plt.title('Number of Iterations on each Partition')

plt.savefig('P2a_hist.png', dpi=200, bbox_inches='tight')