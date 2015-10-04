def extraArgs(neighbors,distance):
	"""If a node is in neighbors, and it hasn't been visited, return it. Else return"""
	def updateNode(n):
		if n[0] in neighbors and n[1] > distance+1:
			# if n[3] == False:
			# 	accum2.add(1)
			if n[3] == False:
				#accum.add(1)
				return (n[0],distance+1,n[2],True)
			else:
				return (n[0],distance+1,n[2],n[3])
		else:
			return n
	return updateNode

def f(neighbors):
	return lambda y:y[3]==False and y[0] in neighbors

def bfs(graph,startNode):	
	print 'Start Node:',startNode
	#greyNodes are the nodes currently being visited. Get the starting node and its neighbors
	greyNodes= set(graph.filter(lambda x:x[0]==startNode).collect())
	for i in range(4):
		#Store neighbors of currently being visted nodes
		neighbors = set([])
		print 'Nodes touch on iteration',i,len(greyNodes)
		#Add the neighbors of each currently being visited node's neighbors to greyNodes
		for x in greyNodes.copy():
			neighbors.update(x[2])
			greyNodes.remove(x)
		#Retrieve the nodes in neighbors
		greyNodes.update(graph.filter(f(neighbors)).collect())
		#Update the distances and visited attribute of nodes in graph
		graph = graph.map(extraArgs(neighbors,i))
		graph.count()
	return graph