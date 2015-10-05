from P2 import *
from random import *

# Draw the figure
def f_random(key):
    return randrange(1,101,1)
    
xy = [(i,j) for j in range(1,2001) for i in range(1,2001)]
arr = sc.parallelize(xy, 1).partitionBy(numPartitions=100, partitionFunc=f_random)
mand = arr.map(lambda x: ((x[0], x[1]),mandelbrot((x[1]/500.0)-2, (x[0]/500.0)-2)))
draw_image(mand)

# Plot the histogram
plt.figure(figsize=(10,10))
plt.hist(sum_values_for_partitions(mand).collect())
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))