from P2 import *

import matplotlib.pyplot as plt
import numpy as np
import random

X = range(2000)
Y = range(2000)
random.shuffle(X) 
random.shuffle(Y)

sc.setLogLevel("WARN")

# create two RDD's of x and y coordinates
x_coords = sc.parallelize(X, 10)
y_coords = sc.parallelize(Y, 10)

# cartesian product of the those two RDD's 
pixels = x_coords.cartesian(y_coords)
mandel = pixels.map(lambda (x,y): ((x,y), mandelbrot(y/ 500. - 2,x/500.-2)))

draw_image(mandel)

plt.hist(sum_values_for_partitions(mandel).collect())
plt.show()
plt.savefig('P2b.jpg')