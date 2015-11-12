from P2 import *

import findspark
findspark.init('/home/toby/spark')

import pyspark
import numpy as np

if __name__ == '__main__':
    sc = pyspark.SparkContext(appName="Spark 2")
    sc.setLogLevel('WARN') 

    x = np.arange(1, 2001)
    x = sc.parallelize(x)

    pixels = x.cartesian(x)
    pixels = pixels.partitionBy(100)
    values = pixels.map(lambda key: [key, mandelbrot(key[1]/500.0-2, key[0]/500.0-2)])

    partitions = sum_values_for_partitions(values).collect()
    plt.hist(partitions, bins=20)
    plt.title("Histogram Of Per-partition Work")
    plt.xlabel("Partitions")
    plt.ylabel("Iterations")
    plt.savefig("P2b_hist.png")

    draw_image(values)
