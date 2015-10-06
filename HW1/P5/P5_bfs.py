'''
Uses BFS to search through a graph starting from the source node. 
	graph: a list, in adjacency list format. Each element in the list is
		(character, [list of adjacent characters])
	source: the name of the first source character to start from
'''
def search(graph, source, end, sc):
	#keep track of an accumulator of how many edges we've explored each time
	#we can terminate when we don't explore any new nodes
	to_explore = sc.accumulator(1)

	#also use an accumulator to keep track of when we hit the target node
	at_target = sc.accumulator(0)

	def explore_node(node):
		next_nodes = []
		name, (neighbors, distance, visited_flag, previous_neighbor) = node
		
		#if the visited flag is 1 (not explored), we want to
		#add each neighbor into the graph with visited flag=1 and distance=distance+1
		if visited_flag == 1:							
			next_nodes += [(neighbor, ([], distance+1, 1, name)) for neighbor in neighbors]

			#also set the visited flag to 2 and return this node back to the graph
			next_nodes.append((name, (neighbors, distance, 2, previous_neighbor)))

			to_explore.add(len(next_nodes))

			if name==end:
				at_target.add(1)

		#if the visited flat is not 1, return this node as is
		else:
			next_nodes.append(node)	

		return next_nodes

	def closest_visited_node(node_a, node_b):		
		neighbors_a, distance_a, visited_flag_a, previous_neighbor_a = node_a
		neighbors_b, distance_b, visited_flag_b, previous_neighbor_b = node_b

		if distance_a < distance_b:
			previous_neighbor = previous_neighbor_a
		else:
			previous_neighbor = previous_neighbor_b

		return (list(set(list(neighbors_a) + list(neighbors_b))), min(distance_a, distance_b), max(visited_flag_a, visited_flag_b), previous_neighbor)

	# Convert the graph of (node, [neighbors]) tuples to
	# (node, ([neighbors], distance, visited_flag, previous_neighbor)) where visited flag is either
	# 	0: not visited, 1: visited but neighbors not explored, or 2: explored
	adjacency_list = graph.map(lambda (x,y): (x, (y, 0, 1, None)) if x==source else (x, (y, 10000, 0, None))).cache()	

	while to_explore.value > 0 and at_target.value == 0:					
		#explore outwards from each node
		to_explore = sc.accumulator(0)
		adjacency_list = adjacency_list.flatMap(explore_node)		
		
		adjacency_list = adjacency_list.reduceByKey(closest_visited_node)
		adjacency_list.count()				
	
	end_point = adjacency_list.lookup(end)[0]	

	return trace_path(adjacency_list, (end, end_point))

#traces back from a node to the start of the path to that node
def trace_path(graph, node, path=[], graph_dict = None):
	if graph_dict == None:
		print "Getting graph"
		graph = graph.filter(lambda (name, (neighbors, distance, visited_flag, previous_neighbor)): distance < 10000).collect()		
		print "Got graph"
		graph_dict = dict([[k,(k,v)] for (k,v) in graph])		

	(current_name, (neighbors, distance, visited_flag, previous_neighbor)) = node

	path.append(current_name)

	if previous_neighbor:		
		previous_node = graph_dict[previous_neighbor]
		#previous_node = graph.filter(lambda (name, (neighbors, distance, visited_flag, previous_neighbor)): name==previous_neighbor).collect()[0]

		return trace_path(graph, previous_node, path, graph_dict)
	else:
		return path

def format_path(reverse_page_list, path):
	formatted_path = [reverse_page_list.lookup(x)[0] for x in path]
	return formatted_path
