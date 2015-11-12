from P2 import *
import time
from  pyspark import SparkContext

sc = SparkContext()
sc.setLogLevel("ERROR")

#Creating the pixel rdd, setting the number of partitions to 10 (later use of cartesian)
pix = sc.parallelize(range(1,2001,1),10)
pixels =  pix.cartesian(pix)

#Sanity check:
pixels.getNumPartitions()

#Transforming the pixel rdd to a coordinate rdd:
xy_rdd = pixels.map(lambda p : (p,((p[1]/500.0)-2,(p[0]/500.0)-2)),True)
xy_rdd.take(10)

#Applying the Mandelbrot function to each value of the xy_rdd
mandel_rdd = xy_rdd.mapValues(lambda val: mandelbrot(val[0],val[1]))

#Draw the image:
#draw_image(mandel_rdd)

#Compute and plot the workload of the different workers:
st = time.time()
to_plot = sum_values_for_partitions(mandel_rdd).collect()
end = time.time()-st
print end

#Around 30s

plt.hist(to_plot)
plt.show()