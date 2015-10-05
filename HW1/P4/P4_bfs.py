import numpy as np
from itertools import chain

def P4_bfs(grph,source,sc):
	
	LoopBreaker = sc.accumulator(0)
	
	nodes = sc.parallelize([(source,0)])
	visitedNodes = sc.parallelize([(source,0)])
	i_dist = 0
	while LoopBreaker.value == 0:
		i_dist+=1
		nodes = grph.join(nodes).values().distinct().partitionBy(8)
		nodes = nodes.mapValues(lambda v: v+1).subtractByKey(visitedNodes)
		if nodes.isEmpty():
			LoopBreaker.add(1)
			continue
		visitedNodes = visitedNodes.union(nodes).cache()
		
	print "Exhausted graph at a distance of:", i_dist
	print "With "+source+" as your source, you touched "+ str(visitedNodes.count()) + " nodes."
	
	return visitedNodes.count(), visitedNodes