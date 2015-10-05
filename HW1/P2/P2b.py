from __future__ import division
import math as ma
import findspark
import os
findspark.init() 
import pyspark
sc = pyspark.SparkContext()

from random import shuffle
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm


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
    plt.title("Mandelbrot Set with Naive Partitions")
    plt.colorbar()
    plt.show()

x_range = range(2000)
y_range = range(2000)
np.random.shuffle(x_range)
np.random.shuffle(y_range)
x_shuffel = sc.parallelize(x_range)
y_shuffel = sc.parallelize(y_range)
rdd_cart_shuffel = x_shuffel.cartesian(y_shuffel)


rdd_mandel_shuffel = rdd_cart_shuffel.map(lambda xy_shuffel: (xy_shuffel, mandelbrot((xy_shuffel[0]/500) - 2, (xy_shuffel[1]/500) - 2)))


rdd_mandel_repart_shuffel = rdd_mandel_shuffel.repartition(100)
draw_image(rdd_mandel_repart_shuffel)


rdd_histogram_shuffel = sum_values_for_partitions(rdd_mandel_repart_shuffel)


hist_data_shuffel = rdd_histogram_shuffel.collect()
ax = plt.subplot()
plt.hist(hist_data_shuffel)
ax.set_xticklabels(hist_data_shuffel, rotation=45)
plt.title("Random Partition Histogram")
plt.show()











