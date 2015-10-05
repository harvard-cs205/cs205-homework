# (c) 2015 L.Spiegelberg
import findspark
findspark.init()

import sys

from P2 import *
from pyspark import SparkContext, SparkConf
import matplotlib.ticker as ticker   
import seaborn as sns
sns.set_style("whitegrid")


# setup spark
conf = SparkConf().setAppName('Mandelbrot')
sc = SparkContext(conf=conf)

# creates a RDD object to pass over to the compute Mandelbrot function
def setupMandelbrot():
    
    # define here count of partitions
    num_partitions = 100;
    
    # create the space
    nx, ny = (2000, 2000)
    x = range(0, nx)
    y = range(0, ny)
    
    rddX = sc.parallelize(x, int(np.sqrt(num_partitions)))
    rddY = sc.parallelize(y, int(np.sqrt(num_partitions)))
    rdd = rddX.cartesian(rddY) # to get tuples
    
    print rdd.getNumPartitions()
    
    return rdd

# given an rdd, this function computes the mandelbrot function
# along with a statistic about iterations per partition
def computeMandelbrot(rdd):
    # now map using mandelbrot to ((I, J), V) with (I, J) being the coordinates
    # and V the value (i.e. iteration number) of the mandelbrot set
    mb = lambda c: (c, mandelbrot((c[0] / 500.0) - 2.0, (c[1] / 500.0) - 2.0))

    # apply mandelbrot function mb to the given set
    rdd = rdd.map(mb)

    # get sum_values_for_partitions and draw a histogram of it
    iteration_stats = sum_values_for_partitions(rdd).collect()
    
    return (rdd, iteration_stats)

# plot a histogram
def plotIterationStats(iteration_stats, savestr):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(np.array(iteration_stats))
    ax.set_xlabel('total number of iterations in million')
    ax.set_ylabel('number of partitions')

    # use the trick from http://stackoverflow.com/questions/10171618/changing-plot-scale-by-a-factor-in-matplotlib
    # to scale figure in a nicer way
    scale = 10e6                                                                                                                                                                                                                                                                  
    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))                                                                                                                                                                                                           
    ax.xaxis.set_major_formatter(ticks) 

    # save figure as P2a_hist.png
    plt.savefig(savestr, dpi=120)


# program logic
def main(argv):
	rdd = setupMandelbrot()

	rdd, iteration_stats = computeMandelbrot(rdd)

	# apply mandelbrot function and draw resulting image
	im = draw_image(rdd)

	plotIterationStats(iteration_stats, 'P2a_hist.png')

# avoid to run code if file is imported
if __name__ == '__main__':
	main(sys.argv)
