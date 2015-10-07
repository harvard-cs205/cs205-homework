"""
Written by Jaemin Cheun
Harvard CS205
Assignment 1
October 6, 2015
"""


# we first assign distances: (node,neighbors) -> (node, (distance, neighbors))
# update the RDD so that it includes information about the distance from the root node. We initialize the distance as 
# infinity for all nodes except for the source, which has a distance of 0
def graph_tranformation(source):
	return (lambda (node, neighbors): (node, (0, neighbors)) if node == source else (node, (float("inf"), neighbors)))

# update the graph so that the distances of next_nodes are updated, but make sure that you update the distance lower (make sure that you have the shortest-path)
def update_graph(depth, next_nodes):
    return (lambda (node, (distance, neighbors)): (node, (min(distance,depth+1), neighbors)) if node in next_nodes else (node, (distance, neighbors)))

def ss_bfs(char_graph, source_char, sc):
	# transforms the graph so that we have distance incorporated.
	char_graph = char_graph.map(graph_tranformation(source_char))
	accumulator = sc.accumulator(0)
	while True:
		depth = accumulator.value
		# we find nodes that has distance of depth, given by the accumulator value
		current_nodes = char_graph.filter(lambda (node, (distance, neighbors)) : distance == depth)
		# if current_nodes is empty, we cannot expand more, so break
		if current_nodes.count() == 0:
			break
		else:
			# we create a list of all possible neighbors of the current nodes, and make sure there are no duplicates by using set
			next_nodes = current_nodes.map(lambda (node, (distance, neighbors)): neighbors).reduce(lambda x,y: list(set(x+y)))
			# update graph
			char_graph = char_graph.map(update_graph(depth, next_nodes))
			accumulator.add(1)
	num_nodes_visited = char_graph.filter(lambda (node, (distance, neighbors)): distance != float('inf')).count()
	print(num_nodes_visited)	


			




