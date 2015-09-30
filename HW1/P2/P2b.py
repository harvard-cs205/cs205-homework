from P2 import *

# Your code here
import pyspark
import numpy as np
import random
import matplotlib.pyplot as plt

sc = pyspark.SparkContext(appName='myAppName')

i = np.arange(1,2001)
j = np.arange(1,2001)

random.shuffle(i)
random.shuffle(j)

i = sc.parallelize(i, 10)
j = sc.parallelize(j, 10)

coordinates = i.cartesian(j)

intensity = coordinates.map(lambda x: [x, mandelbrot(x[1]/500.0-2, x[0]/500.0-2)])

# draw_image(intensity)

plt.hist(sum_values_for_partitions(intensity).collect())
plt.show()
plt.savefig('histgram.png')