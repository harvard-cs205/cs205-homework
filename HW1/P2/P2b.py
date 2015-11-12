from P2 import *
import findspark
findspark.init()
import os
import pyspark
sc = pyspark.SparkContext()

# define shuffled axis: to ensure randomness, the axes are defined separately
axis1 = range(2000)
np.random.shuffle(axis1)
axis2 = range(2000)
np.random.shuffle(axis2)
randomaxis1 = sc.parallelize(axis1,10)
randomaxis2 = sc.parallelize(axis2,10)
randomrdd = randomaxis1.cartesian(randomaxis2)

# other parts of the code remains the same
drawrandomrdd = randomrdd.map(lambda pixel: [pixel, mandelbrot(pixel[1]/500.0-2, pixel[0]/500.0-2)])
draw_image(drawrandomrdd)
plt.hist(sum_values_for_partitions(drawrandomrdd).collect())
plt.savefig('P2b_hist.png')