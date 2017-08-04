from P2 import *
# Your code here

import numpy
import matplotlib.pyplot as plt
import random
# Your code here
# 2000 x 2000 pixels

sc.setLogLevel('ERROR')
X = range(2000)
Y = range(2000)
random.shuffle(X)
random.shuffle(Y)


xs = sc.parallelize(X,10)
ys = sc.parallelize(Y,10)

pixels = xs.cartesian(ys)
mandel = pixels.map(lambda (x,y): ((x,y),mandelbrot((y/500.0) - 2,(x/500.0) - 2)))

#draw_image(mandel)

amt = sum_values_for_partitions(mandel).collect()

plt.figure()
plt.hist(amt)
plt.ylabel('Partitions')
plt.xlabel('Work Done')
plt.savefig('P2b_hist.png') 