# undebugged; conceptual stuff down. Did not run on AWS 
def bfs(node, dest, graph, sc):

	expanded = sc.accumulator(1)

	prev_expanded = 0

	# starting with root which has with depth 0, and the node that lead to it; its backtrance
	nodes = sc.parallelize([(node, (0,node))])

	#depth count
	i = 0

	found = False
	while ((not found) and (prev_expanded < expanded.value)):
		print '---'
		print prev_expanded 
		print expanded

		# add depth value
		i = i +1
		prev_expanded = expanded.value

		# get all next links of the nodes, then store as (depth, cur page, prev page) tuple
		neighbors = nodes.join(graph).flatMap(lambda (K,V) : (V[1],K))

		# add depth to each (page, prevpage) tuple
		neighbors = neighbors.map(lambda x : ((x[0],x[1]),i))

		# join the pages already visted with new pages
		new_n = neighbors.leftOuterJoin(nodes)

		# get new nodes
		new_nodes = new_n.filter(lambda (K,V) : V[1] == None).map(lambda (K,V) : (K,V[0])).cache()

		# update accumulator
		new_nodes.foreach(lambda x: expanded.add(1))

		# if found the end of the path
		if new_nodes.filter(lambda x : x[0] == dest).count() > 0:
			found = True

		# save new nodes and paths
		nodes = nodes.union(new_nodes).cache()

		# to prod lazy execution
		print nodes.count()
	
	# if found, need to backtrack
	if found:
		global nodes
		path = []
		# while not at first node
		while not path[-1] == node:
			# prob doesn't work, but in essence need this functionality
			# look up value of the prev node to this node
			path = path.append(nodes.lookup(path[-1])[1])
		print "found path:"
		return path

	else:
		print "not found"
		return False

