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

    # Declare our SparkContext
    sc = SparkContext("local", "madelbrot")

    # Get the i, j pixel arrays
    xs = sc.parallelize(range(0, NX), 10)
    ys = sc.parallelize(range(0, NY), 10)

    # Take the cartesian product to get the pixel grid
    # Now we partition into 100 partitions, and to determine where each element
    # of the RDD goes, we simply assign a random number between 1 and 100 (inclusive)
    pixels = xs.cartesian(ys).partitionBy(100, lambda p: randint(1, 101) )

    # Now calculate the mandelbrot counts and store with the original pixels
    pixels_and_counts = pixels.map(lambda (x, y): ((x, y), mandelbrot(pixelize(x), pixelize(y))))

    # And generate our histogram
    partition_work = sum_values_for_partitions(pixels_and_counts)
    plt.hist(partition_work.take(partition_work.count()), bins=100)
    plt.xlabel('Amount of Work')
    plt.ylabel('Number of Partitions')
    plt.title('Compute Distribution - Random Partitioning Scheme')
    plt.savefig('part_b.png')

    draw_image(pixels_and_counts)
    
