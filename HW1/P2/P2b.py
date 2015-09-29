# Initialize SC context
import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext(appName="Spark 1")
from P2 import *

# Your code here
if __name__ == '__main__':
  # Set up Initial RDDs
  rdd = sc.parallelize(range(2000),10)
  rdd1 = sc.parallelize(range(2000),10)
  rdd2 = rdd1.cartesian(rdd)
  # Generate RDD for (K,V) = ((i,j),mandelbrot())
  rdd2_new = rdd2.partitionBy(100)
  rdd3_new = rdd2_new.map(lambda (i,j): ((i,j), mandelbrot((j/500.0)-2,(i/500.0)-2)))
  # Draw a mandelbrot plot
  rdd4_new = draw_image(rdd3_new)
  # Save a sequence of per-partition efforts
  rdd5_new = sum_values_for_partitions(rdd3_new)
  seq_of_efforts_new = rdd5_new.collect()
  # Plot a histrogram
  #ind = np.arange(len(seq_of_efforts_new))
  #plt.bar(ind+0.2,seq_of_efforts_new,0.6)
  plt.hist(seq_of_efforts_new)
  plt.savefig("P2b_hist.png")
