import numpy as np
from itertools import chain

def P4_bfs(grph,source,sc):
	
	nodes = sc.parallelize([(source,0)])
	visitedNodes = sc.parallelize([(source,0)])
	i_dist = 0
	while not nodes.isEmpty():
		nodes = grph.join(nodes).values().distinct().mapValues(lambda v: v+1).partitionBy(20)
		nodes = nodes.subtractByKey(visitedNodes)
		visitedNodes = visitedNodes.union(nodes).partitionBy(20).cache()
		i_dist+=1
		
	print "Exhausted graph at a distance of:", i_dist
	print "With "+source+" as your source, you touched "+ str(visitedNodes.count()) + " nodes."
	
	return visitedNodes.count(), visitedNodes