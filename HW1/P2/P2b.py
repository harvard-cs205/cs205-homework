from P2 import *
import random
# File used with Spark on Shell

# Building the data
data = xrange(1, 2000)
rdd = sc.parallelize(data, 10)
combinations = rdd.cartesian(rdd)

# Mapping the pixel (i,j) to the value: mandelbrot (x_i, x_j)
combinations = rdd.cartesian(rdd)\
    .map(lambda x: (x, mandelbrot(x[1]/500.0-2, x[0]/500.0-2)))\
    .partitionBy(100, lambda x: random.randrange(0, 100))
combinations_cos = combinations\
    .map(lambda x: (x, mandelbrot(x[1]/500.0-2, x[0]/500.0-2)))\
    .partitionBy(100, lambda x: x[0] / (np.sqrt(x[0]**2 + x[1]**2)))

# Drawing the image
draw_image(combinations)


# Ploting the computation effort on each partition: the value represent the
# the number of iterations.
computation_effort = sum_values_for_partitions(combinations_cos).collect()

plt.hist(computation_effort)
plt.xlabel('Computation effort')
plt.ylabel('Number of Partitions')
plt.show()
