from P2 import *
from pyspark import SparkContext

# Your code here

#Method 1
# for i in range(2000):
# 	for j in range(2000):
# 		mat.append((i,j))
# #		mat.append(((j/500.0)-2,(i/500.0)-2))
# #		tmp.append(((j/500.0)-2,(i/500.0)-2))
# sc = SparkContext()
# m = sc.parallelize(mat,1200)
# # mandelResult = m.map(lambda t:[t,mandelbrot(t[0],t[1])]).filter(lambda a: a[1]!=0).count()
# mandelResult = m.map(lambda t:[t,mandelbrot((t[0]/500.0)-2,(t[1]/500.0)-2)])

# print sum_values_for_partitions(mandelResult).collect()
#draw_image(mandelResult)
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
# pdb.set_trace()
# for a in mandelResult.collect():
# 	if a[1]!=0:
# 		print 'NOT ZERO!!!!!!!!!!!!'
# 	break
#Do we need tuples in matrix format???
# sc = SparkContext()
# a=sc.parallelize([1,2,3,4])
# b=a.map(lambda x:x**2)
# print b.take(10)
