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

x = sc.parallelize(range(2000))
y = sc.parallelize(range(2000))
rdd_cart = x.cartesian(y)

rdd_mandel = rdd_cart.map(lambda xy: (xy, mandelbrot((xy[0]/500) - 2, (xy[1]/500) - 2)))

rdd_mandel_repart = rdd_mandel.repartition(100)
draw_image(rdd_mandel_repart)

rdd_histogram = sum_values_for_partitions(rdd_mandel_repart)


hist_data = rdd_histogram.collect()
ax = plt.subplot()
plt.hist(hist_data)
ax.set_xticklabels(hist_data, rotation=45)
plt.title("Naive Partition Histogram")
plt.show()






