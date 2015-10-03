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
		
		#-----------------------------------------------
		#Trace backwards through the graph, originating with dest and collect the nodes connecting it to the source
		destDist = np.min(visitedNodes.lookup(dest))
		traceDist = destDist #Count the dist we're tracing backward
		nodeTraceback = [dest]
		path = []
		while traceDist > 0:
			path_tuple = visitedNodes.join(grph).filter(lambda x: x[1][0] == traceDist).filter(lambda x: x[0] in nodeTraceback).collect()[0]
			nodeTraceback = path_tuple[1][1]
			path_tuple = (path_tuple[0], path_tuple[1][0])
			traceDist -= 1 #Decrement counter
			path.append(path_tuple)
		path.reverse()
		#----------------------------------------------------
		return destDist, visitedNodes.count(), path
	else:
		return vistedNodes, visitedNodes.count()