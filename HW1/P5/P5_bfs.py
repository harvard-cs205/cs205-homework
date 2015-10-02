import numpy as np

def P5_bfs(grph,source,sc,dest=False):
	sourceNode = [(source,0)]
	print sourceNode, type(sourceNode)
	nodes = sc.parallelize(sourceNode)
	visitedNodes = sc.parallelize(sourceNode)
	
	while not nodes.isEmpty():
		nodes = grph.join(nodes).flatMapValues(lambda x: [(t,x[1]+1) for t in x[0]]).values().distinct()
		nodes = nodes.subtractByKey(visitedNodes)
		visitedNodes = visitedNodes.union(nodes).cache()
		
		if not visitedNodes.filter(lambda x: x[0]==dest).isEmpty():
			print "Found destination!"
			break
	
	visitedNodes = visitedNodes.groupByKey().mapValues(list)
	
	
	if dest:
		return np.min(visitedNodes.lookup(dest)), visitedNodes.count()
	else:
		return vistedNodes, visitedNodes.count()