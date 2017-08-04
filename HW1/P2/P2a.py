from P2 import *

import numpy
import matplotlib.pyplot as plt

# Your code here
# 2000 x 2000 pixels
sc.setLogLevel('ERROR')
X = 2000
Y = 2000

xs = sc.parallelize(range(X),10)
ys = sc.parallelize(range(Y),10)

pixels = xs.cartesian(ys)
mandel = pixels.map(lambda (x,y): ((x,y),mandelbrot((y/500.0) - 2,(x/500.0) - 2)))

#draw_image(mandel)

amt = sum_values_for_partitions(mandel).collect()

plt.figure()
plt.hist(amt)
plt.ylabel('Partitions')
plt.xlabel('Work Done')
plt.savefig('P2a_hist.png')