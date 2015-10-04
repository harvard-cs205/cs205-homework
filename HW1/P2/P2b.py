from P2 import *
from  pyspark import SparkContext
import time

sc = SparkContext()
sc.setLogLevel("ERROR")

#Creating the pixel rdd
pix2 = sc.parallelize(np.random.permutation(range(1,2001,1)),10)
pix3 = sc.parallelize(np.random.permutation(range(1,2001,1)),10)
pixels2 =  pix2.cartesian(pix3)

xy_rdd2 = pixels2.map(lambda p : (p,((p[1]/500.0)-2,(p[0]/500.0)-2)))
print xy_rdd2.getNumPartitions()
mandel_rdd2 = xy_rdd2.mapValues(lambda val : mandelbrot(val[0],val[1]))

#draw_image(mandel_rdd2)
st = time.time()
to_plot = sum_values_for_partitions(mandel_rdd2).collect()
end = time.time()-st

print end 
# Around 50s
plt.hist(to_plot,bins=20)
plt.show()

