from P2 import *
import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext(appName="Spark1")
my_rdd = sc.parallelize(xrange(2000),10)
rdd = my_rdd.cartesian(my_rdd)
ans = rdd.map(lambda x: [x,mandelbrot((x[1]/500.0-2),(x[0]/500.0-2))])
draw_image(ans)

ans1 = sum_values_for_partitions(ans)

plt.hist(ans1.collect())
plt.savefig('P2a_hist.png')
