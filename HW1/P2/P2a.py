from P2 import *

import findspark
findspark.find()
findspark.init(edit_profile=True)
import pyspark
sc = pyspark.SparkContext()

def mandelbrot(x, y):
    z = c = complex(x, y)
    iteration = 0
    max_iteration = 511  # arbitrary cutoff
    while abs(z) < 2 and iteration < max_iteration:
        z = z * z + c
        iteration += 1
    return iteration

# Your code here
def form():
    iteration_count = []
    for i in range(20):
        for j in range(20):
            x = j/5 - 2
            y = i/5 - 2
            iteration_count.append(mandelbrot(x, y))
    return iteration_count


rdd = sc.parallelize()
