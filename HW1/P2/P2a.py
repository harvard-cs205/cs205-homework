import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext()

from P2 import *

x = np.arange(2000)
y = np.arange(2000)
grid = sc.parallelize(x, 10).cartesian(sc.parallelize(y, 10))
mandelbrot_grid = grid.map(lambda p: (p, mandelbrot(p[0]/500.0 - 2, p[1]/500.0 - 2)))

draw_image(mandelbrot_grid)

