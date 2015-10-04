from P2 import *
import numpy as np
import matplotlib.pyplot as plt
import pyspark


if __name__ == "__main__":
    sc = pyspark.SparkContext()

    i_rdd = sc.parallelize(xrange(2000), 10)
    j_rdd = sc.parallelize(xrange(2000), 10)
    complex_rdd = i_rdd.cartesian(j_rdd).cache()
    result = complex_rdd.map(lambda x: ((x[0], x[1]),mandelbrot(x[1]/500.0 -2, x[0]/500.0 -2)))

    draw_image(result)

    sum_rdd = sum_values_for_partitions(result)
    plt.hist(sum_rdd.collect(), 10)
    plt.xlabel('Number of computation')
    plt.ylabel('Percentage')
    plt.title('Computation Distribution')    
    plt.savefig('test_hist_a.png')