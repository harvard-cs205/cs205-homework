from pyspark import SparkContext
from pyspark.accumulators import AccumulatorParam
import sys

# Flags
UNVISITED = 0
VISITING = 1
VISITED = 2

# Class for accumulator
class BFSAccumulator(AccumulatorParam):
	def zero(self, val):
		return {}

	# Get union of keys and set value as min distance from source node
	def addInPlace(self, accum1, accum2):
		return { k: min(i for i in (accum1.get(k), accum2.get(k)) if i != None) for k in accum1.viewkeys() | accum2}


# If node is a set as a VISITNG node, explode A by generating a list of A's immediate neighbors as 
# tuples that get flatmapped to the rdd, with their distances from the source set as A's distance+1.
def explode_nodes(node, accum):
	character = node[0]
	info = node[1]

	# If we aren't visiting this node or if this node has nothing to visit, don't change info
	if info[2] != VISITING:
		return [(character, info)]

	# Explode nodes
	nodes = []
	neighbors = list(info[0])
	for neighbor in neighbors:
		node = (neighbor, (set(), info[1]+1, VISITING))
		nodes.append(node)

	# Append current node's info, and set it as visited
	current_node = (character, (info[0], info[1], VISITED))
	nodes.append(current_node)

	# Add to the accumulator to indicate that a node with edges has been touched
	accum.add({character: info[1]})

	return nodes

# Clean up mess made by explode_nodes. Combine multiple information tuples (edges, distance, and flag) of a node as so:
# 	1) Grab a valid set of edges
#  	2) Grab the smallest distance from the source
# 	3) Grab the flag that represents that max value (listed above)
def combine_nodes(value1, value2):
	neighbors1 = value1[0]
	neighbors2 = value2[0]
	dist1 = value1[1]
	dist2 = value2[1]
	flag1 = value1[2]
	flag2 = value2[2]

	# Combine node values
	neighbors = neighbors1 if len(neighbors1) > len(neighbors2) else neighbors2
	dist = min(dist1, dist2)
	flag = max(flag1, flag2)
	
	return (neighbors, dist, flag)

# Set init distances and flags for nodes
def init_nodes(node, source):
	character = node[0]
	edges = node[1]

	if character == source:
		return (character, (edges, 0, VISITING))

	return (character, (edges, sys.maxint, UNVISITED))

# Performs bfs on an rdd of edges by setting an arbitrarily high distance from the source
# node and sets flags depending on if the node is unvisited, visiting, or visited.
# At each step, a visiting node A is exploded into a list of nodes, whose distances are
# A's distance from the source node + 1. A is flagged as a visited node, and the exploded nodes
# are grouped with their corresponding nodes.
def bfs(graph, source, sc):
	# Define accumulator and initial value for nodes touched
	accum = sc.accumulator({}, BFSAccumulator())
	nodes_touched = 0

	# Set distances from source node
	distances = graph.map(lambda node: init_nodes(node, source))

	#	Explode and combine the nodes until nodes are unreachable
	while True:
		distances = distances.flatMap(lambda x: explode_nodes(x, accum)) \
								.reduceByKey(combine_nodes) \
								.filter(lambda (node, info): info[2] != VISITED) \
								.cache()
		
		# Apply transformations
		distances.count()

		# If accumulator didn't touch more nodes, BFS is complete
		if len(accum.value) - nodes_touched == 0:
			break

		# Record nodes touched in previous iteration
		nodes_touched = len(accum.value)

	return accum.value