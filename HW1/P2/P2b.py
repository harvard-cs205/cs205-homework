import numpy as np
from P2 import *

import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext()

x = np.random.permutation(2000)
y = np.random.permutation(2000)
grid = sc.parallelize(x, 10).cartesian(sc.parallelize(y, 10))
image = grid.map(lambda (x, y): ((x, y), mandelbrot((x / 500.) - 2, (y / 500.) - 2)))
draw_image(image)

partition_values = sum_values_for_partitions(image).collect()
plt.hist(partition_values)
plt.xlabel('Number of Iterations')
plt.ylabel('Number of Partitions')
plt.title('Distribution of Partitions by Number of Iterations')
plt.show()
