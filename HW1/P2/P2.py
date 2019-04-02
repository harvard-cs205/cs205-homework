import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
#from __future__ import division

def quiet_logs(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)
    logger.LogManager.getLogger("amazonaws").setLevel(logger.Level.WARN)


def mandelbrot(x, y):
    z = c = complex(x, y)
    iteration = 0
    max_iteration = 511  # arbitrary cutoff
    while abs(z) < 2 and iteration < max_iteration:
        z = z * z + c
        iteration += 1
    return iteration

def sum_values_for_partitions(rdd):
 #   'Returns (as an RDD) the sum of V for each partition of a (K, V) RDD'
    # note that the function passed to mapPartitions should return a sequence,
    # not a value.
    return rdd.mapPartitions(lambda part: [sum(V for K, V in part)])

def draw_image(rdd):
#    '''Given a (K, V) RDD with K = (I, J) and V = count,
#    display an image of count at each I, J'''
    data = rdd.collect()
    #
    I = np.array([d[0][0] for d in data])
    J = np.array([d[0][1] for d in data])
    C = np.array([d[1] for d in data])
    #
    im = np.zeros((I.max() + 1, J.max() + 1))
    im[I, J] = np.log(C + 1)  # log intensity makes it easier to see levels
    plt.imshow(im, cmap=cm.gray)
    plt.show()
    plt.savefig("Mandelbrot", dpi=2000, facecolor='w', edgecolor='w')

q = range(0,4000)
X = sc.parallelize(q,100)
coordinates = X.flatMap(lambda x: [(x,y) for y in q]) # successfully creates pixel coordinates...

data = coordinates.map(lambda x: (x,mandelbrot((x[1]/1000.-2),(x[0]/1000.-2))))
draw_image(data)

#draw_image(data) # this does indeed draw the mandelbrot set. takes a while.

# sum_values_for_partitions doesn't work: Exception: It appears that you are attempting to broadcast an RDD or reference an RDD from an action or transformation. RDD transformations and actions can only be invoked by the driver, not inside of other transformations; for example, rdd1.map(lambda x: rdd2.values.count() * x) is invalid because the values transformation and count action cannot be performed inside of the rdd1.map transformation. For more information, see SPARK-5063.
# can't seem to make it work.

#test test test