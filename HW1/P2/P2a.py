from P2 import *
import findspark
import matplotlib.pyplot as plt
findspark.init()
import pyspark
sc = pyspark.SparkContext()

# Your code here
if __name__ == "__main__":
    init_rdd = sc.parallelize(range(0,2000),10)
    mand_rdd = init_rdd.cartesian(init_rdd)
    mand_rdd = mand_rdd.map(lambda x: (x,mandelbrot((float(x[1])/500)-2, (float(x[0])/500)-2)))
    partition_val = sum_values_for_partitions(mand_rdd).collect()
    plt.hist(partition_val)
    plt.xlabel("No. of Computations")
    plt.ylabel("No. of Partitions")
    plt.title("Histogram for distribution of computations over partitions")
    plt.savefig("P2a.png")
