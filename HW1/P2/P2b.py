from P2 import *

# Your code here
sc=SparkContext()
xrand=range(2000)
yrand=range(2000)
random.shuffle(xrand)
random.shuffle(yrand)
x = sc.parallelize(xrand,10)
y = sc.parallelize(yrand,10)
xy = x.cartesian(y)
result = xy.map(lambda t:[t,mandelbrot((t[0]/500.0)-2,(t[1]/500.0)-2)])
draw_image(result)
y_values = sum_values_for_partitions(result).collect()
plt.hist(y_values)
plt.savefig('P2b.png')