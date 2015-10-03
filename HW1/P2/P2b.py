from P2 import *
from pyspark import SparkContext
import random

if __name__ == "__main__":
    # initialize SparkContext
    sc = SparkContext(appName="Mandelbrot P2b")
    
    # create a vector from 0-1999 (2000 elements) across 10 partitions
    vector = sc.parallelize(range(0,2000),10)
    
    # create 2000 x 2000 coordinates across 10 x 10 = 100 partitions, then partition each coordinate across the 100 partitions randomly
    coords = vector.cartesian(vector).partitionBy(100, lambda coord: random.randint(0,99))
    
    # create a tuple of the form ((x, y), intensity) as determined by the mandelbrot function
    # note: cannot use same variable names as functions (e.g., mandelbrot)
    mandelRDD = coords.map(lambda coord: (coord, mandelbrot(coord[1]/500.0-2,coord[0]/500.0-2)))

    # visualize the results (if desired)
    #draw_image(mandelRDD)
    
    # determine the amount of "compute" for each partition
    h = sum_values_for_partitions(mandelRDD).collect()
    
    # create a bar chart containing the compute at each partition
    ind = np.arange(len(h))
    plt.figure()
    plt.bar(ind, h)
    plt.ylabel('Compute')
    plt.xlabel('Partitions')
    plt.savefig("P2b_hist.png")
    
    # create a histogram of the distribution of compute amounts across all partitions
    plt.figure()
    plt.hist(h)
    plt.ylabel('Number of Partitions')
    plt.xlabel('Compute')
    plt.savefig("P2b_hist2.png")
