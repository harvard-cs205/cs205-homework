# Spark code implementing breadth first search, given a graph RDD and a character name, using the optimized 
# version from the last part.

def bfs(graph, start):
	# Queue keeps track of nodes to be explored at each level of BFS
	q = [start]

	# Distance dictionary keeps track of all distances of visited/seen nodes
	dist = {}
	dist[start] = 0
	distFromSource = 0
	diameter = 10

	while q and distFromSource < diameter:
		# Finds nodes that are in search queue
		nodesToExplore = graph.filter(lambda (k,v): k in q)

		# Finds its neighbors
		neighborSet = reduce(set.union, nodesToExplore.values().collect())
		q = list(neighborSet.difference(dist.keys()))

		# Update distances of already visited nodes 
		distFromSource += 1
		for node in q:
			dist[node] = distFromSource

	return dist