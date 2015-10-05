# (c) 2015 L.Spiegelberg
# provides code for Problem 4
from pyspark import SparkContext, SparkConf

import csv
import pandas as pd
import os
import sys
import urllib

from P4_bfs import *

# lookup code
def lookUpVertexID(dictRDD, character):
    return dictRDD.filter(lambda x: x[0] == character).collect()[0][1]

def lookUpCharacter(dictRDD, vertexID):
    return dictRDD.filter(lambda x: x[1] == vertexID).collect()[0][0]

# program logic
def main(argv):
	# setup spark
	conf = SparkConf().setAppName('Graph Processing')
	sc = SparkContext(conf=conf, pyFiles=['P4_bfs.py', 'source.csv'])

	sc.setLogLevel('ERROR')

	filename = 'source.csv'

	# Conversion from MarvelGraph to adj. list representation
	comicRDD = sc.textFile(filename, 32).map(lambda x: x.split('","')).map(lambda x: (x[0][1:], x[1][:len(x[1])-1]))

	# create Dict (last part can be removed as it does only make indices 1-based)
	dictRDD = comicRDD.map(lambda x: x[0]).distinct().sortBy(lambda x: x).zipWithIndex().map(lambda x: (x[0], x[1]+1))

	# convert Character to integer for later join!
	comicRDD = comicRDD.join(dictRDD).map(lambda x: (x[1][0], x[1][1])).cache()

	# join on Comic Issue & remove all reflexive edges
	graphRDD = comicRDD.join(comicRDD).map(lambda x: (x[1][0], x[1][1])).filter(lambda x: x[0] != x[1])

	# now group s.t. we have for each vertex an adjacency list of nodes
	graphRDD = graphRDD.groupByKey().map(lambda x: (x[0], list(x[1])))


	# nodes to search for
	v0 = [lookUpVertexID(dictRDD,'CAPTAIN AMERICA'), \
	   	  lookUpVertexID(dictRDD,'MISS THING/MARY'), \
	      lookUpVertexID(dictRDD,'ORWELL')]

	# write results to output.txt
	with open('output.txt', 'wb') as f:
		for i in range(0, len(v0)):
		    rdd = loadEdgeList(sc, filename)
		    num_visited_nodes, rdd = sparkBFS(sc, rdd, v0[i])
		    
		    message = '%s : %d nodes visited' % (lookUpCharacter(dictRDD, str(v0[i])), num_visited_nodes)
		    print(message)
		    f.write(message + '\n')


# avoid to run code if file is imported
if __name__ == '__main__':
	main(sys.argv)
