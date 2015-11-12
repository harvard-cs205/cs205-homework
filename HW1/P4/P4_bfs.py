from pyspark import AccumulatorParam

# Set custom accumulator class for sets
class AccumulatorParamSet(AccumulatorParam):
	def zero(self, initialValue):
		return set()
	def addInPlace(self, v1, v2):
		v1 |= v2
		return v1

# Conduct breadth first search on graph with source node start
def bfs(graph, start, sc):
	# Queue keeps track of nodes to be explored at each level of BFS
	q = set([start])

	# Visited set keeps track of all visited nodes so far
	visited = sc.accumulator(set(), AccumulatorParamSet())

	while q:
		# Finds nodes that need to be explored
		prevVisited = set(list(visited.value))
		graph.filter(lambda (k,v): k in q).foreach(lambda (k,v): visited.add(v))

		# New nodes that need to be explored in next iteration
		q = visited.value - prevVisited

	return visited.value