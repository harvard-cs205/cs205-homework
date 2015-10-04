# (c) 2015 L.Spiegelberg
import findspark
findspark.init()

import sys
import numpy as np
from scipy.optimize import curve_fit

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
    
    # create the space (this can be also done using spark's cartesian command)
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

# low res mandelbrot for sigmoid partitioning
def runLowResMandelbrot():
	# define here count of partitions
	num_partitions = 4;

	# create the space 
	nx, ny = (100, 100)
	x = range(0, nx)
	y = range(0, ny)

	rddX = sc.parallelize(x, int(np.sqrt(num_partitions)))
	rddY = sc.parallelize(y, int(np.sqrt(num_partitions)))
	rdd = rddX.cartesian(rddY) # to get tuples

	mb = lambda c: (c, mandelbrot((c[0] / 25.0) - 2.0, (c[1] / 25.0) - 2.0))
	rdd = rdd.map(mb)

	return draw_image(rdd)
    
    

# program logic
def main(argv):
	rdd = setupMandelbrot()


	# change partitions here manually for better load balancing
	num_partitions = rdd.getNumPartitions()

	## Strategy 1: Random Partitioning
	# define a custom partitioner function
	def randomPartition(key): 

	    # return a partition id 0, ..., num_partitions-1 random
	    return np.random.randint(num_partitions)

	rdd = setupMandelbrot()

	# see https://spark.apache.org/docs/1.1.1/api/python/pyspark.rdd.RDD-class.html#partitionBy
	rdd = rdd.partitionBy(num_partitions, partitionFunc=randomPartition)

	rdd, iteration_stats = computeMandelbrot(rdd)
	plotIterationStats(iteration_stats, 'P2b_hist.png')

	## Strategy 2: Sigmoid partitioning
	def sigmoid(x, x0, k):
		y = 1 / (1 + np.exp(-k*(x-x0)))
		return y

	def invsigmoid(y, x0, k):
		x = np.log(y / (1-y)) / k + x0
		return x

	# run first lower resolution mandelbrot
	Imlow = runLowResMandelbrot()

	# retrieve from image load (=sum of colors) vs . x
	C = np.exp(Imlow) -1
	Cy = np.sum(C, axis=1)
	Cy = np.cumsum(Cy) / np.sum(C)
	Cx = range(0, C.shape[0])

	# fit sigmoid function to data
	popt, pcov = curve_fit(sigmoid, Cx, Cy)

	# now split the image space [0, 1] into num_partitions buckets and
	# retrieve size of bucket via inv of the sigmoid function
	sizes = [invsigmoid((i * 1.0 + 0.5)  / num_partitions, *popt) 
	for i in range(0, num_partitions)]
	sizes[99] = num_partitions

	# as we only want to split x (avoid subsplitting y,
	# would be also possible), make sure each partition is at least 1 big
	# scale sizes now to resolution
	nx = 2000
	sizes = [int(np.ceil(nx * x / num_partitions)) for x in sizes]

	# define using this size vector the partitoning function
	def sigmoidPartition(key): 
	    
	    # key is a pixel position (x, y) with x = 0, ..., n_x -1
	    x = key
	    
	    # return a partition id 0, ..., num_partitions-1 
	    # based on sizes vector
	    partition = 0
	    for size in sizes:
	    	if x < size:
	    		break
	    	partition += 1

	    return partition

	# run exoeriment
	rdd = setupMandelbrot()
	rdd = rdd.partitionBy(num_partitions, partitionFunc=sigmoidPartition)
	rdd, iteration_stats = computeMandelbrot(rdd)
	plotIterationStats(iteration_stats, 'P2b_hist_sigmoid.png')


# avoid to run code if file is imported
if __name__ == '__main__':
	main(sys.argv)
