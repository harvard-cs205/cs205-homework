# (c) 2015 L.Spiegelberg

import findspark
findspark.init()
from pyspark import SparkContext, SparkConf


import pandas as pd
# compute shortest path!
from P5_sssp import *

def main(argv):
	# setup spark
	conf = SparkConf().setAppName('WikiGraph')
	sc = SparkContext(conf=conf, pyFiles=['P5_sssp.py'])

	# number of partitions (for AWS 32)
	num_partitions = 4

	# paths to the big data files
	datapath = '../../../../../courses/CS205_Computing_Foundations/data/'
	titlespath = datapath + 'titles-sorted.txt'
	linkspath = datapath + 'links-simple-sorted.txt'

	rddTitles = sc.textFile(titlespath, num_partitions)
	rddGraph = sc.textFile(linkspath, num_partitions)

	# prepare Graph into adjacency list structure
	rddGraph = rddGraph.map(lambda x: x.replace(':', '').split(' ')) \
					   .map(lambda x: (int(x[0]), [int(y) for y in x[1:]]))

	rddTitles = rddTitles.zipWithIndex().map(lambda x: (x[0], x[1] + 1)).cache()

	# lookup a title (its index)
	keyString0 = 'Kevin_Bacon'
	keyStringT = 'Harvard_University'
	v0 = rddTitles.filter(lambda x: x[0] == keyString0).collect()[0][1]
	vT = rddTitles.filter(lambda x: x[0] == keyStringT).collect()[0][1] 

	# run bfs
	num_visited_nodes, rddPaths = sparkSSSP(sc, rddGraph, v0, vT, 20)
	# run bfs for reverse direction
	num_visited_nodesR, rddPathsR = sparkSSSP(sc, rddGraph, vT, v0, 20)

	# collect paths and save to file (if something happens,
	# we can recover here as comoutation needs some time)
	resultPaths = rddPaths.collect() + rddPathsR.collect()

	with open('paths.txt', 'wb') as f:
		f.write(str(resultPaths))

	# parallelize again, to make Spark perform the dictionary lookup

	rddLookup = sc.parallelize(resultPaths)

	# for better performance on join, 
	rddLookup.cache()
	nodesOfInterest = rddLookup.flatMap(lambda x: x).distinct().collect()
	selectedTitles = rddTitles.filter(lambda x: x[1] in nodesOfInterest).map(lambda x: (x[1], x[0]))

	resultPaths = rddLookup.zipWithIndex() \
	  .flatMap(lambda x: [(y, (x[1]+1, i)) for i, y in enumerate(x[0], 1)]) \
	  .join(selectedTitles) \
	  .map(lambda x: (x[1][0][0], (x[1][0][1], x[1][1]))) \
	  .groupByKey().map(lambda x: zip(*sorted(list(x[1])))[1]).collect()

	 #backup
	with open('output.txt', 'wb') as f:
		f.write(str(resultPaths))

	# collect the paths in a human readable format
	with open('outputH.txt', 'wb') as f:
		f.write('Shortest paths from %s to %s:\n\n' % (keyString0, keyStringT))
		for path in resultPaths:
	            path = list(path) 
	            f.write(path[0])
	            for node in path[1:]:
	                f.write(' -> ' + node)
	            f.write('\n')

# avoid to run code if file is imported
if __name__ == '__main__':
	main(sys.argv)
         