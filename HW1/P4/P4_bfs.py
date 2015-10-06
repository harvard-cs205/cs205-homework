'''
Uses BFS to search through a graph starting from the source node. 
	graph: a list, in adjacency list format. Each element in the list is
		(character, [list of adjacent characters])
	source: the name of the first source character to start from
'''
def search(graph, source, sc):
	#keep track of an accumulator of how many edges we've explored each time
	#we can terminate when we don't explore any new nodes
	to_explore = sc.accumulator(1)

	def explore_node(node):
		next_nodes = []
		name, (neighbors, distance, visited_flag) = node

		#if the visited flag is 1 (not explored), we want to
		#add each neighbor into the graph with visited flag=1 and distance=distance+1
		if visited_flag == 1:						
			next_nodes += [(neighbor, ([], distance+1, 1)) for neighbor in neighbors]

			#also set the visited flag to 2 and return this node back to the graph
			next_nodes.append((name, (neighbors, distance, 2)))

			#increment the counter so that we know there are new neighbors which we've explored
			to_explore.add(len(next_nodes))
		
		#if the visited flat is not 1, return this node as is
		else:
			next_nodes.append(node)	

		return next_nodes

	def closest_visited_node(node_a, node_b):		
		neighbors_a, distance_a, visited_flag_a = node_a
		neighbors_b, distance_b, visited_flag_b = node_b

		return (list(set(list(neighbors_a) + list(neighbors_b))), min(distance_a, distance_b), max(visited_flag_a, visited_flag_b))

	# Convert the graph of (node, [neighbors]) tuples to
	# (node, ([neighbors], distance, visited_flag)) where visited flag is either
	# 	0: not visited, 1: visited but neighbors not explored, or 2: explored
	adjacency_list = graph.map(lambda (x,y): (x, (y, 0, 1)) if x==source else (x, (y, 10000, 0)) )	

	while to_explore.value > 0:		
		#explore outwards from each node
		to_explore = sc.accumulator(0)
		adjacency_list = adjacency_list.flatMap(explore_node)

		#reduce to find the 'correct' value for duplicate nodes, and force evaluation
		adjacency_list = adjacency_list.reduceByKey(closest_visited_node)
		adjacency_list.count()

	#filters down to the list of nodes which we've touched
	touched = adjacency_list.filter(lambda (name, (neighbors, distance, visited_flag)): distance < 10000)
	print touched.count()

	#finds the maximum distance
	counts = touched.collect()
	print max(counts, key=lambda (name, (neighbors, distance, visited_flag)): distance)

	return adjacency_list
