
def bfs(node, graph, sc):
	expanded = sc.accumulator(1)

	prev_expanded = 0



	# starting with root which has with depth 0
	nodes = sc.parallelize([(node,0)])

	#depth count
	i = 0

	while (prev_expanded < expanded.value):
		# print '---'
		# print prev_expanded 
		# print expanded

		# increase depth
		i = i +1
		prev_expanded = expanded.value
		# get all distinct neighbors of the nodes
		neighbors = nodes.join(graph).flatMap(lambda (K,V) : V[1]).distinct()

		# add depth value
		neighbors = neighbors.map(lambda x : (x,i))

		# combine to get new neighbors to consider
		new_n = neighbors.leftOuterJoin(nodes)

		# filter for only new nodes
		new_nodes = new_n.filter(lambda (K,V) : V[1] == None).map(lambda (K,V) : (K,V[0])).cache()

		# update Accum
		new_nodes.foreach(lambda x: expanded.add(1))

		# add to explored nodes
		nodes = nodes.union(new_nodes).cache()

		# force evaluation
		ret = nodes.count()
		print ret
	return nodes.count()
