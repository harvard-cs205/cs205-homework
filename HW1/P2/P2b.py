import pyspark
from pyspark import SparkContext
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
sc = SparkContext("local")


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
    
#Perform same steps as in P2a, however now we call repartition to randomly
#repartition RDD into 100 partitions. note, i was forced to call .partitionBy
#first for some reasons or else repartition would not seem to work.
    
pixel = range(2000)
pix1D=sc.parallelize(pixel,10)
pixels = pix1D.cartesian(pix1D).partitionBy(100).repartition(100)

#no need to draw in this step


mandrdd = pixels.map(lambda i: (i, mandelbrot((i[1]/500.)-2,(i[0]/500.0)-2)))

plt.hist(sum_values_for_partitions(mandrdd).collect())
plt.savefig("P2b_hist.png")
plt.xlabel("Iteration Count")
plt.ylabel("Count")
