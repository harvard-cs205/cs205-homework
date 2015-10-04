# (c) 2015 L.Spiegelberg
# this file contains code, to automatically find a shortest path between two given nodes
# and outputs all possible paths as a list to output.txt
import findspark
findspark.init()
from pyspark import SparkContext, SparkConf
import matplotlib.ticker as ticker   
import pandas as pd

import seaborn as sns
sns.set_style('whitegrid')

# shortest path code is in this module
from P5_sssp import *

# setup spark
conf = SparkConf().setAppName('WikiGraph')
sc = SparkContext(conf=conf, pyFiles=['P5_sssp.py'])
sc.setLogLevel('ERROR')

# function to prepare two rdds, one holding the graph, the other for later use as a dictionary
def prepareWikiGraph(titlefile, linksfile):

	rddTitles = sc.textFile(titlefile)
	rddGraph = sc.textFile(linksfile)

	# title file has structure v: v0, ...., v_d
	# simple mapping will give the rdd structure
	# (v, [v_1, ..., v_d]) as needed by the sssp algorithm
	rddGraph = rddGraph.map(lambda x: x.replace(':', '').split(' ')) \
					   .map(lambda x: (int(x[0]), [int(y) for y in x[1:]]))

	# note that for the wikigraph everything is 1-indexed
	# dictionary has structure ('a wiki title', 23)
	rddTitles = rddTitles.zipWithIndex().map(lambda x: (x[0], x[1] + 1)).cache()
	return rddGraph, rddTitles

# helper function to lookup given the titles dictionary a vertex
def vertexByTitle(rddTitles, keyString):

	# lookup a title (its index)
	# (if there are more, use only first result --> passed rdd should be a dict!)
	return rddTitles.filter(lambda x: x[0] == keyString).collect()[0][1]

# helper function to convert sssp output to readable paths
# returns a list of tuples, where each tuple represents a graph
def convertPathsToHuman(rddPaths, rddTitles):
	# for better performance on join, filter the large dictionary
	# therefore extract all the important nodes!
	rddPaths.cache()
	nodesOfInterest = rddPaths.flatMap(lambda x: x).distinct().collect()
	selectedTitles = rddTitles.filter(lambda x: x[1] in nodesOfInterest).map(lambda x: (x[1], x[0]))

	pathList = rddPaths.zipWithIndex() \
	                    .flatMap(lambda x: [(y, (x[1]+1, i)) for i, y in enumerate(x[0], 1)]) \
	                    .join(selectedTitles) \
	                    .map(lambda x: (x[1][0][0], (x[1][0][1], x[1][1]))) \
	                    .groupByKey().map(lambda x: zip(*sorted(list(x[1])))[1]).collect()
         
	return pathList

# main function holding all the functionality
def main(argv):

	# paths to the big data files
	datapath = '../../../../../courses/CS205_Computing_Foundations/data/'
	titlespath = datapath + 'titles-sorted.txt'
	linkspath = datapath + 'links-simple-sorted.txt'

	# setup initial rdds
	rddGraph, rddTitles = prepareWikiGraph(titlespath, linkspath)

	# cache the graph
	rddGraph = rddGraph.cache()

	startTitle  = 'Kevin_Bacon'
	targetTitle = 'Harvard_University'

	v0 = vertexByTitle(rddTitles, startTitle)
	vT = vertexByTitle(rddTitles, targetTitle)


	# call shortest paths (limit iterations to 20)
	max_iterations = 20

	num_visited_nodes, rddPaths = sparkSSSP(sc, rddGraph, v0, vT, max_iterations)

	# convert paths to human readable format
	shortestPaths = convertPathsToHuman(rddPaths, rddTitles)

	# write output to file
	with open('output.txt', 'wb') as f:
		f.write('Shortest paths from %s to %s:\n\n' % (startTitle, targetTitle))
		for path in shortestPaths:
	            path = list(path) 
	            f.write(path[0])
	            for node in path[1:]:
	                f.write(' -> ' + node)
	            f.write('\n')

# avoid to run code if file is imported
if __name__ == '__main__':
	main(sys.argv)


