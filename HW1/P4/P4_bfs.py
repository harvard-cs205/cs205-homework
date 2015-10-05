
def reset_accumulators(sc):
	sc.unvisited = sc.accumulator(0)
	sc.explored = sc.accumulator(0)

def make_to_be_visited(explored, unvisited):
	"""Create to_be_visited function with given accumulators"""
	def to_be_visited((n, (adj, distance))):
		"""Check if a node has been visited after join"""
		if distance is not None:
			unvisited.add(1)
			return True
		explored.add(1)
		return False
	return to_be_visited


def reduce_distances((n, (d1, d2))):
	return (n, d2) if d2 is not None else (n, d1)

def bfs2(sc, adjacencies, root):
	"""Uses accumulators to relax graph diameter assumption"""
	reset_accumulators(sc)
	to_visit = sc.parallelize([(root, 0)])
	distances = sc.parallelize([])
	last_visited_state = (-1, -1) 

	i = 1
	while(True):
		last_unvisited = sc.unvisited.value
		last_explored = sc.explored.value

		visited_this_iteration = adjacencies.leftOuterJoin(to_visit).filter(make_to_be_visited(sc.explored, sc.unvisited))
		visited_this_iteration.count() # force evaluation so accumulators update
		flattened = visited_this_iteration.map(lambda (n, (adj, distance)): (n, distance))
		distances = distances.fullOuterJoin(flattened).map(reduce_distances)
		to_visit = visited_this_iteration.flatMap(lambda (n, (adj, distance)): adj).distinct().map(lambda a:(a, i))

		#  terminate if the number of unvisited and visited nodes is unchanged: this means the bfs hasn't made progress
		visited_state = (sc.unvisited.value - last_unvisited, sc.explored.value - last_explored)
		if visited_state == last_visited_state:
			break
		else:
			last_visited_state = visited_state

		i += 1

	return distances.count()

def bfs1(sc, adjacencies, root, graph_diameter=10):
	"""BFS on adjacency matrix from root, returns number of nodes explored. Requires graph diameter"""

	inf = float('inf')
	set_initial_distance = lambda (n, adj) : (n, 0 if n == root else inf)
	distances = adjacencies.map(set_initial_distance)

	queue = set([root])
	for i in range(1, graph_diameter+1):
		to_process = lambda (n, adj): n in queue
		to_add = lambda (n, adj): adj
		queue = set(adjacencies.filter(to_process).flatMap(to_add).collect())
		distances = distances.map(lambda (n, d): (n, min(d, i)) if n in queue else (n, d))
		distances = sc.parallelize(distances.collect())

	explored = lambda (n, distance): distance < inf
	n_nodes_touched = len(distances.filter(explored).collect())

	return n_nodes_touched