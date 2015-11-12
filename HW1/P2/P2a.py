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

    # Declare our SparkContext
    sc = SparkContext("local", "mandelbrot")

    # Get the i, j pixel arrays
    # Make sure they have 100 partitions (cartesian multiplies the partitions of each)
    xs = sc.parallelize(range(0, NX), 10)
    ys = sc.parallelize(range(0, NY), 10)

    # Take the cartesian product to get the pixel grid
    pixels = xs.cartesian(ys)

    # Now calculate the mandelbrot counts and store with the original pixels
    pixels_and_counts = pixels.map(lambda (x, y): ((x, y), mandelbrot(pixelize(x), pixelize(y))))
    
    # Generate plots to demonstrate work per partitions
    partition_work = sum_values_for_partitions(pixels_and_counts)
    plt.hist(partition_work.take(partition_work.count()), bins=100)
    plt.xlabel('Amount of Work')
    plt.ylabel('Number of Partitions')
    plt.title('Compute Distribution - Naive Partitioning Scheme')
    plt.savefig('part_a.png')

    draw_image(pixels_and_counts)
    
