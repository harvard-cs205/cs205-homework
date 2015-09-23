from P2 import *
from pyspark import SparkContext
import numpy as np
import matplotlib.pyplot as plt
from random import randint

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
    pixels = xs.cartesian(ys).partitionBy(100, lambda p: randint(1, 101) )

    # Now calculate the mandelbrot counts and store with the original pixels
    pixels_and_counts = pixels.map(lambda (x, y): ((x, y), mandelbrot(pixelize(x), pixelize(y))))
    partition_work = sum_values_for_partitions(pixels_and_counts)
    plt.hist(partition_work.take(partition_work.count()))
    plt.savefig('test_hist_b.png')

    draw_image(pixels_and_counts)
    
