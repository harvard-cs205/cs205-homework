import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
#from pyspark import SparkContext


def mandelbrot(x, y):
    z = c = complex(x, y)
    iteration = 0
    max_iteration = 511  # arbitrary cutoff
    while abs(z) < 2 and iteration < max_iteration:
        z = z * z + c
        iteration += 1
    return iteration
def dummy(x,y):
	return (x+y)*5
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

if __name__ == "__main__":
  #  sc = SparkContext(appName="Mandelbrot")
    Nmax=2000
    idx=sc.parallelize(range(1,Nmax),10) #desired number of partitions is 100, 
    xy=idx.cartesian(idx).map(lambda i: ((i[1],i[0]),(i[1]/500.0-2,i[0]/500.0-2)))
    z=xy.map(lambda i: (i[0], mandelbrot(i[1][1],i[1][0])))
    draw_image(z)
    
    partition_counts=sum_values_for_partitions(z).collect()
    
    