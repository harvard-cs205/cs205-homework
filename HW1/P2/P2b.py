from P2 import *
import numpy as np
import matplotlib.pyplot as plt
import findspark
findspark.find()
findspark.init('/usr/local/opt/apache-spark/libexec')
import pyspark
# Your code here
# Modification 1
sc = pyspark.SparkContext()
    
i_rdd = sc.parallelize(xrange(2000), 10)
j_rdd = sc.parallelize(xrange(2000), 10)

complex_rdd = i_rdd.cartesian(j_rdd).partitionBy(100, lambda hash_key: np.random.randint(0, 100, size=1)[0])
result = complex_rdd.map(lambda x: ((x[0], x[1]),mandelbrot(x[1]/500.0 -2, x[0]/500.0 -2)))
sum_rdd = sum_values_for_partitions(result)

plt.hist(sum_rdd.collect(), 10)
plt.xlabel('Number of computations for each partition ')
plt.ylabel('Percentage')
plt.title('Computation Distribution')	
plt.savefig('P2b_hist.png')
