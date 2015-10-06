from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyspark

def mandelbrot(x, y):
    z = c = complex(x, y)
    iteration = 0
    max_iteration = 511  # arbitrary cutoff
    while abs(z) < 2 and iteration < max_iteration:
        z = z * z + c
        iteration += 1
    return iteration

def sum_values_for_partitions(rdd):
    'Returns (as an RDD) the sum of V for each partition of a (K, V) RDD'
    # note that the function passed to mapPartitions should return a sequence,
    # not a value.
    return rdd.mapPartitions(lambda part: [sum(V for K, V in part)])

def draw_image(rdd):
    '''Given a (K, V) RDD with K = (I, J) and V = count,
    display an image of count at each I, J'''

    data = rdd.collect()
    I = np.array([d[0][0] for d in data])
    J = np.array([d[0][1] for d in data])
    C = np.array([d[1] for d in data])
    im = np.zeros((I.max() + 1, J.max() + 1))
    im[I, J] = np.log(C + 1)  # log intensity makes it easier to see levels
    plt.imshow(im, cmap=cm.gray)
    plt.show()

def draw_partitions(rdd):
    parts = rdd.mapPartitionsWithIndex(lambda idx, part: [(K, np.exp(idx)) for K, V in part])
    draw_image(parts)

if __name__ == "__main__":
    # initialize spark context
    sc = pyspark.SparkContext("local[4]", "Spark1")

    # create RDD of pixel location tuples
    i_vals = sc.parallelize(range(2000), 10)
    ij_vals = i_vals.cartesian(i_vals)

    # associate mandelbrot values with each pixel
    ijxy_vals = ij_vals.zip(ij_vals).mapValues(lambda (i,j): (j/500 - 2, i/500 - 2))
    mandel_vals = ijxy_vals.mapValues(lambda xy: mandelbrot(*xy))
    #draw_image(mandel_vals)
    #draw_partitions(mandel_vals)

    plt.hist(sum_values_for_partitions(mandel_vals).collect(), bins=50)
    plt.xlabel('iteration steps')
    plt.ylabel('partitions')
    plt.savefig('P2a_hist.png')
