import findspark
findspark.init()
import pyspark

from P2 import *

def main(): 
    sc = pyspark.SparkContext()
    sc.setLogLevel("WARN")
    x = np.arange(2000)
    y = np.arange(2000)
    # will use 100 partitions by default
    grid = sc.parallelize(x, 10).cartesian(sc.parallelize(y, 10))
    mandelbrot_grid = grid.map(lambda p: (p, mandelbrot(p[0]/500.0 - 2, p[1]/500.0 - 2)))
    h = sum_values_for_partitions(mandelbrot_grid)
    plt.hist(h.collect())
    plt.title("Default Partitioning Work")
    plt.xlabel("Partition Number")
    plt.ylabel("Work")
    plt.savefig("P2a_hist.png")
    plt.show()
    #draw_image(mandelbrot_grid)
    
if __name__=="__main__":
    main()

