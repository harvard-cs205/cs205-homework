%matplotlib inline
import findspark
findspark.init()
import pyspark

# shut down the previous spark context
sc.stop() 
sc = pyspark.SparkContext(appName="myAppName")
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import pdb
import time
import math
import random 

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
    return I,J
    

# Create the pixels
i = np.array(xrange(0,2000))
j = i

def Pixel_values(i,j):
    x = j/500.00 - 2.0
    y = i/500.0 - 2.0
    count = mandelbrot(x,y)
    return (i,j),count
 

pi = sc.parallelize(range(2000))
pj = sc.parallelize(range(2000))

# Create the tuples K =  (i,j)

# Here we can sort by distance to the point that requires the most iterations
#K = pi.cartesian(pj).sortBy(lambda x: math.sqrt( (x[0]-1000.0)**2 + (x[1]-1000.0)**2  ), ascending = True)
#K = K.map(lambda x :(math.sqrt( (x[0]-1000.0)**2 + (x[1]-1000.0)**2  ),(x[0],x[1]) ))

# Here we redistribute randomly the K tuplets 
K = pi.cartesian(pj).sortBy(lambda x: random.randint(0,1) , ascending = True)

# Now check the distance to the point that required the most iteration
K = K.partitionBy(100).map(lambda x :(math.sqrt( (x[0]-1000.0)**2 + (x[1]-1000.0)**2  ),(x[0],x[1]) ))


# collect the partitions into a list and check the distance to the point: 1000 , 1000

#myK = K.glom().collect()

# Print the first partition

#myK[-1]

# Now apply the function mantelbrot to the pixel
rdd = K.partitionBy(100).map(lambda p : Pixel_values(p[1][0], p[1][1]))

t = sum_values_for_partitions(rdd).collect()

plt.hist(t,bins=np.linspace(1e6,1e7)  );
#plt.xscale('log')
plt.xlim([1e6,3*1e6])
plt.xlabel('Number of iterations', fontsize = 20 )
plt.ylabel('Number of partitions',fontsize = 20)
plt.grid(True)
plt.gcf().set_size_inches(8, 8)
plt.savefig('P2a hist.png', box = 'tight')
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.savefig('P2b_hist.png')