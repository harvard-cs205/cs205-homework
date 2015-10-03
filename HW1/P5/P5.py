# Your code here
#import pyspark
from pyspark import SparkContext
from P5_bfs import *

if __name__ == "__main__":
	sc = SparkContext("local", "Simple App")

    # make spark shut the hell up
	logger = sc._jvm.org.apache.log4j
	logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
	logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

	import numpy as np 
	import itertools

	linklist = sc.textFile('links-simple-sorted.txt')
	titlelist = sc.textFile('titles_sorted.txt')
	numerical_titles = titlelist.zipWithIndex()
	print numerical_titles.take(10)
	split_list = linklist.map(lambda x: x.split(':'))
	print split_list.take(10)

	# Create an RDD of directed edges, and subtract 1 from every edge to make it 0 indexed.
	adj = split_list.flatMapValues(lambda x: x.split()).map(lambda (n1, n2): (int(n1) -1 , int(n2) -1))
	#adj = split_list.mapValues(lambda x: x.split()).map(lambda (str_x, str_neighbors): (int(str_x) - 1, map(lambda x: int(x) - 1, str_neighbors)))
	#adj = split_list.flatMapValues(lambda x: x.split()).map(lambda (n1, n2): (int(n1) -1 , int(n2) -1)).groupByKey()

	print adj.take(10)
	#print edges.take(100)
	#start = "Kevin_Bacon"
	#end = "Harvard_University"
	start = 'TITLE_a'
	end = "TITLE_e"

	start_node = numerical_titles.lookup(start)[0]
	end_node = numerical_titles.lookup(end)[0]
	print start_node, end_node

	distances, unreachables = bfs(adj, start_node, sc)

	print "Distance to end node:", distances.lookup(end_node)[0]
