from P2 import *
from pyspark import SparkContext


#Method 2 Cartesian of [0,2000] and [0,2000]
sc=SparkContext()
x=range(2000)
y=range(2000)
x = sc.parallelize(x,10)
y = sc.parallelize(y,10)
xy = x.cartesian(y)
result = xy.map(lambda t:[t,mandelbrot((t[0]/500.0)-2,(t[1]/500.0)-2)])
draw_image(result)
y_values = sum_values_for_partitions(result).collect()
plt.hist(y_values)
plt.savefig('P2a.png')
#plt.show()

