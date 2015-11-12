from P2 import *
from pyspark import SparkContext

# Set up partitions and context
N = 2000
sc = SparkContext("local", "P2a")

# Partition a 10x10 cartesian grid of workers
rdd = sc.parallelize(range(N), 10)
plot = rdd.cartesian(rdd)

# Apply Mandelbrot function to each square and draw the fractal
mb_image = plot.map(lambda x: (x, mandelbrot((x[0]/500.0-2), (x[1]/500.0-2))))
draw_image(mb_image)

# Plot the histogram using provided function
mb_data = sum_values_for_partitions(mb_image)
mb_plot = mb_data.collect()
plt.hist(mb_plot)
plt.xlabel("Partition")
plt.ylabel("Number of Iterations")
plt.title("Standard Sequential Partition Histogram")
plt.savefig("P2a_hist.png")