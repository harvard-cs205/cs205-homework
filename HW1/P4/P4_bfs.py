import Queue

def nonRecursiveBFS(Graph,startingNode):
	q = Queue.Queue()
	Graph.node[startingNode]['distance'] = 0
	q.put(startingNode)
	currentMax=0
	print 'Starting'
	while not q.empty():
		node = q.get()
		nodeDist = Graph.node[node]['distance']
		# print currentMax,nodeDist
		currentMax = nodeDist if nodeDist > currentMax else currentMax
		for c in Graph.neighbors(node):
			if Graph.node[c]['distance'] == -1:
				Graph.node[c]['distance'] = Graph.node[node]['distance']+1
				q.put(c)
	return currentMax

def resetDistance(Graph):
	for n in Graph.nodes():
		Graph.node[n]['distance']=-1