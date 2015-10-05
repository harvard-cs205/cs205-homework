import pyspark
from pyspark import SparkContext
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import random
sc = SparkContext("local")
sc.setLogLevel("ERROR")

#import functions from HW
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

#define single array of 2000 pixels and use .cartesian() to get 2 dimensions.
#partition size of 10 is initially used because .cartesian results in 10x10 = 100    
pixel = range(2000)
pix1D=sc.parallelize(pixel,10)
pixels = pix1D.cartesian(pix1D)
mandrdd = pixels.map(lambda i: (i, mandelbrot((i[1]/500.)-2,(i[0]/500.0)-2)))
draw_image(mandrdd)
plt.hist(sum_values_for_partitions(mandrdd).collect(),bins=6)
plt.xlabel("Iteration Count")
plt.ylabel("Count")
plt.savefig("P2a_hist.png")