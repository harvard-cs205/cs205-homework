from P2 import *
import numpy as np
import matplotlib.pyplot as plt
from pyspark import SparkContext

# Convert pixel (i, j) to the correct (x, y) value
def pixelize(x):
    return x / 500. - 2

# Number of x and y pixels
NX = NY = 2000

if __name__ == "__main__":
    sc = SparkContext("local", "madelbrot")

    # Get the i, j pixel arrays
    xs = sc.parallelize(range(0, NX), 10)#.map(pixelize)
    ys = sc.parallelize(range(0, NY), 10)#.map(pixelize)

    # Take the cartesian product to get the pixel grid
    # Also add a count variable to each
    pixels = xs.cartesian(ys)

    # Now calculate the mandelbrot counts and store with the original pixels
    pixels_and_counts = pixels.map(lambda (x, y): ((x, y), mandelbrot(pixelize(x), pixelize(y))))

    partition_work = sum_values_for_partitions(pixels_and_counts)
    plt.hist(partition_work.take(partition_work.count()), bins=100)
    plt.xlabel('Partition Number')
    plt.ylabel('Compute')
    plt.title('Total Compute vs. Partition - Naive Partitioning Scheme')
    plt.savefig('test_hist_a.png')

    draw_image(pixels_and_counts)
    
