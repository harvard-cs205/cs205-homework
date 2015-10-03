# Initialize SC context
import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext(appName="Spark 1")
from P2 import *

# Your code here
if __name__ == '__main__':
  # Set up Initial RDDs with 100 partitions
  rdd = sc.parallelize(range(2000),10)
  rdd1 = sc.parallelize(range(2000),10)
  rdd2 = rdd1.cartesian(rdd)

  # Generate RDD for (K,V) = ((i,j),mandelbrot())
  rdd3 = rdd2.map(lambda (i,j): ((i,j), mandelbrot((j/500.0)-2,(i/500.0)-2)))
  # Draw a mandelbrot plot (Only for Plot)
  #rdd4 = draw_image(rdd3)

  # Save a sequence of per-partition efforts
  rdd5 = sum_values_for_partitions(rdd3)
  seq_of_efforts = rdd5.collect()

  # Plot a histrogram (Also plot a bar)
  ## For histrogram
  #plt.hist(seq_of_efforts)
  #plt.savefig("P2a_hist.png")
  # For Bar plot
  ind = np.arange(len(seq_of_efforts))
  plt.bar(ind+0.2,seq_of_efforts,0.6)
  plt.savefig("P2a_bar.png")

  
