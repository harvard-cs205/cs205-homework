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

    manded = distr_z.map(lambda (a,b): ((a,b), mandelbrot((b/500. - 2),(a/500. - 2))))
    # Create K,V pairs with the xy coordinates as the K, and the mandelbrot value as the V

    #draw_image(manded)
    #    Pretty picture!!

    effort = sum_values_for_partitions(manded).collect()
    # Collect the sum of the number of iterations performed within each parition by calling
    #     the helper function

    endTime=time.time()

    plt.hist(effort, bins=10)
    plt.title("Effort, default partitioning, took " + str(endTime-startTime) + " seconds")
    plt.show()
