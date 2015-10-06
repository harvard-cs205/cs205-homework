from P2 import *
from pyspark import SparkContext

N = 2000

# Your code here
sc = SparkContext("local", "P2a")

rdd = sc.parallelize(range(N), 10)
plot = rdd.cartesian(rdd)

mb_image = plot.map(lambda x: (x, mandelbrot((x[0]/500.0-2), (x[1]/500.0-2))))
draw_image(mb_image)

mb_data = sum_values_for_partitions(mb_image)
mb_plot = mb_data.collect()
plt.hist(mb_plot)
plt.xlabel("Partition")
plt.ylabel("Number of Iterations")
plt.title("Standard Sequential Partition Histogram")
plt.savefig("P2a_hist.png")