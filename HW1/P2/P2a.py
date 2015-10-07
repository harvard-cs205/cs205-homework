from P2 import *
import findspark
findspark.init()
import pyspark

# Your code here
def main():
    sc = pyspark.SparkContext()
    arr = [i for i in xrange(2000)]
    rdd = sc.parallelize( arr, 10 )
    rdd = rdd.cartesian( rdd )
    rdd.cache()
    rdd_xy = rdd.map( lambda ( i, j ): ( ( i, j ), ( (float(j)/500) - 2, (float(i)/500) - 2 ) ) )
    rdd_xy.cache()
    rdd_mandelbrot = rdd_xy.map( lambda ( x, y ): ( x, mandelbrot( y[0], y[1] ) ) )
    rdd_mandelbrot.cache()
    effort_per_partition = sum_values_for_partitions(rdd_mandelbrot)
    effort_per_partition.cache()
    effort_per_partition_collect = effort_per_partition.collect()
    plt.hist( effort_per_partition_collect )
    plt.savefig( "P2a_hist" )
    plt.show()
    draw_image( rdd_mandelbrot )

if __name__ == "__main__": main()