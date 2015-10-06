from P2 import *

import matplotlib.pyplot as plt
import numpy as np

X = 2000 
Y = 2000

sc.setLogLevel("WARN")

# create two RDD's of x and y coordinates
x_coords = sc.parallelize(range(X), 10)
y_coords = sc.parallelize(range(Y), 10)

# cartesian product of the those two RDD's 
pixels = x_coords.cartesian(y_coords)
mandel = pixels.map(lambda (x,y): ((x,y), mandelbrot(x/ 500. - 2,y/500.-2)))

draw_image(mandel)







