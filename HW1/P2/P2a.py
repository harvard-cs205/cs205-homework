import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext(appName="Spark1")
import matplotlib.pyplot as plt 
import time

def mandelbrot(x, y):
    z = c = complex(x, y)
    iteration = 0
    max_iteration = 511  # arbitrary cutoff
    while abs(z) < 2 and iteration < max_iteration:
        z = z * z + c
        iteration += 1
    return iteration

def mandelspark(dim=2000):
	'''Creates coordinate matrix rdd with serial partitioning, maps mandelbrot values'''
	vec1   = vec2 = sc.parallelize(xrange(dim),10)
	coords = vec1.cartesian(vec2) # serial block partitions by coordinate grid
	rdd    = coords.map(lambda tup: (tup, mandelbrot( tup[1]/500.-2., tup[0]/500.-2.)))
	return rdd

def mandelclock(f):
	'''Performance testing'''
	T1 = time.time()
	rdd = f()
	T2 = time.time()
	dT = T2-T1
	print "elapsed time:", dT
	return rdd

def mandelplot(data,title,fname,bins=30):
	'''Plot histogram, save to file'''
	plt.hist(data,bins)
	plt.ylim([0,90]) # consistent x/y axis bounds for both plots
	plt.xlim([0,2.5e7]) 
	plt.title(title)
	plt.savefig(fname)

def sum_values_for_partitions(rdd):
    '''Returns (as an RDD) the sum of V for each partition of a (K, V) RDD'''
    # note that the function passed to mapPartitions should return a sequence,
    # not a value.
    return rdd.mapPartitions(lambda part: [sum(V for K, V in part)])

figtitle = "Sequential partitioning (2a)"
figname  = "P2a_hist.png"
data = sum_values_for_partitions(mandelclock(mandelspark)).collect()

mandelplot(data, figtitle, figname)
