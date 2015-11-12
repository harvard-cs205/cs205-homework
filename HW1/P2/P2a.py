from P2 import *
# File used with Spark on Shell

# Building the data
data = xrange(1, 2000)
rdd = sc.parallelize(data, 10)
combinations = rdd.cartesian(rdd)

# Mapping the pixel (i,j) to the value: mandelbrot (x_i, x_j)
combinations = combinations.map(lambda x: (x, mandelbrot(x[1]/500.0-2,
                                x[0]/500.0-2)))

draw_image(combinations)


# Ploting the computation effort on each partition: the value represent the
# the number of iterations.
computation_effort = sum_values_for_partitions(combinations).collect()

plt.hist(computation_effort)
plt.xlabel('Computation effort')
plt.ylabel('Number of Partitions')
plt.show()
