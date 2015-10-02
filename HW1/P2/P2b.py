from P2 import *
import findspark
findspark.init()
import pyspark
import matplotlib.pyplot as plt
import time
 
if __name__ == "__main__":  
    startTime=time.time()
    
    sc = pyspark.SparkContext()

    ran = range(0,2000, 1)
    z = [(a,b) for a in ran for b in ran]
    # a list of the pairs of x,y values to plot
    
    distr_z = sc.parallelize(z, 100)
    # split z up into 100 partitions

    distr_z_1 = distr_z.repartition(100)
    # randomly repartition into 100 partitions

    manded = distr_z_1.map(lambda (a,b): ((a,b), mandelbrot((b/500. - 2),(a/500. - 2))))

    #draw_image(manded)
    #    Pretty picture!!

    effort = sum_values_for_partitions(manded).collect()
    # Collect the sum of the number of iterations performed within each parition by calling
    #     the helper function

    endTime=time.time()

    plt.hist(effort, bins=10)
    plt.title("Effort, randomized partitioning, took " + str(endTime-startTime) + " seconds")
    plt.xlim(0,9000000)
    plt.show()


### A few things below that were failed experiments.  I don't want to delete them in case
###     I might like to come back to them later.  The gist of the below approach was to create
###     100 partitions that are each composed of points that are evenly spaced across the 
###     4 million points (rather than taken randomly).  
   
""" 
    #ranges = [map(lambda x: x+i, range(0,2000,10)) for i in range(0,10,1)]
    
    #ranges = sc.parallelize(range(0,20,1))

    ranges = sc.parallelize([map(lambda x: x+i, range(0,100,10)) for i in range(0,10,1)], 10)
    
    bal = sc.parallelize(range(0,10,1), 1)

    def zipr(x): return x.cartesian(x)
    print bal.foreachPartition(zipr).collect()
    
    #print bal.cartesian(bal).collect()

    #rangesRDD = sc.parallelize(ranges,10)
    #partitions100 = rangesRDD.cartesian(rangesRDD)
    #print partitions100.collect()
"""

"""
    ran = range(0,2000, 1)
    z = [(a,b) for a in ran for b in ran]
    # a list of the pairs of x,y values to plot
    
    distr_z = sc.parallelize(z, 100)
    # split z up into 100 partitions

    manded = distr_z.map(lambda (a,b): ((a,b), mandelbrot((b/500. - 2),(a/500. - 2))))
    # Create K,V pairs with the xy coordinates as the K, and the mandelbrot value as the V

    #draw_image(manded)
    #    Pretty picture!!

    effort = sum_values_for_partitions(manded).collect()
    # Collect the sum of the number of iterations performed within each parition by calling
    #     the helper function

    plt.hist(effort, bins =100)
    plt.title("Effort, default partitioning")
    plt.show()
"""
