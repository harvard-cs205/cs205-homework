from pyspark import SparkContext
from pyspark.accumulators import AccumulatorParam
import sys
from P5_bfs import *

############################################################################
######################################BFS
############################################################################

# Flags
UNVISITED = 0
VISITING = 1
VISITED = 2

# Class for accumulator
class BFSAccumulator(AccumulatorParam):
	def zero(self, val):
		return {}

	# Get union of keys and set value as min distance from source node and retrieve proper parent
	def addInPlace(self, accum1, accum2):
		accum = {}

		for k in dict(accum1, **accum2):
			value1 = accum1.get(k)
			value2 = accum2.get(k)

			if value1 == None:
				distance = value2[0]
				parent = value2[1]
			elif value2 == None:
				distance = value1[0]
				parent = value1[1]
			else:
				if value1[0] == value2[0]:
					distance = value1[0]
					parent = value1[1] or value2[1]
				elif value1[0] < value2[0]:
					distance = value1[0]
					parent = value1[1]
				else:
					distance = value2[0]
					parent = value2[1]

			accum[k] = (distance, parent)

		return accum

# If node is a set as a VISITNG node, explode A by generating a list of A's immediate neighbors as 
# tuples that get flatmapped to the rdd, with their distances from the source set as A's distance+1.
def explode_nodes(node, accum=None, target=None):
	character = node[0]
	info = node[1]

	# If we aren't visiting this node or if this node has nothing to visit, don't change info
	if info[2] != VISITING:
		return [(character, info)]

	# Append current node's info, and set it as visited
	nodes = []
	current_node = (character, (info[0], info[1], VISITED, info[3]))
	nodes.append(current_node)

	# Add to the accumulator to indicate that a node with edges has been touched
	accum.add({character: (info[1], info[3])})

	# Explode nodes
	neighbors = list(info[0])
	for neighbor in neighbors:
		# Expand current neighbor
		node = (neighbor, ([], info[1]+1, VISITING, character))
		nodes.append(node)

		# If we find target, then return nodes
		if neighbor == target:
			accum.add({neighbor: (info[1]+1, character)})
			return nodes

	return nodes

# Clean up mess made by explode_nodes. Combine multiple information tuples (edges, distance, and flag) of a node as so:
# 	1) Grab a valid set of edges (ignored if empty set)
#  	2) Grab the smallest distance from the source
# 	3) Grab the flag that represents that max value (listed above)
#		4) Grab parent of node with shorter distance from source
def reduce_nodes(info1, info2):
	neighbors1 = info1[0]
	neighbors2 = info2[0]
	dist1 = info1[1]
	dist2 = info2[1]
	flag1 = info1[2]
	flag2 = info2[2]
	parent1 = info1[3]
	parent2 = info2[3]

	# Combine node info
	neighbors = neighbors1 if len(neighbors1) > len(neighbors2) else neighbors2
	if dist1 == dist2:
		dist = dist1
		if parent1 != None:
			parent = parent2
		else:
			parent = parent1
	elif dist1 < dist2:
		parent = parent1
		dist = dist1
	else:
		parent = parent2
		dist = dist2
	flag = max(flag1, flag2)
	
	return (neighbors, dist, flag, parent)

# Set init distances, flags, and parent for nodes
def init_nodes(node, source):
	page_id = node[0]
	edges = node[1]

	if page_id == source:
		return (page_id, (edges, 0, VISITING, None))

	return (page_id, (edges, sys.maxint, UNVISITED, None))

# Return path by tracing to source node in accumulated node dictionary
def print_nodes(prev, target):
	path = []
	current = target
	while current != None:
		path.append(current)
		current = prev[current][1]
	return list(reversed(path))

# Performs bfs on an rdd of edges by setting an arbitrarily high distance from the source
# node and sets flags depending on if the node is unvisited, visiting, or visited.
# At each step, a visiting node A is exploded into a list of nodes, whose distances are
# A's distance from the source node + 1. A is flagged as a visited node, and the exploded nodes
# are grouped with their corresponding nodes. Return if we have reached target.
def bfs(sc,graph, source, target):
	# Define accumulator
	accum = sc.accumulator({source: (0, None)}, BFSAccumulator())

	# Set distances from source node
	print 'init ORIGINAL GRAPH'
	distances = graph.map(lambda node: init_nodes(node, source))

	#	Explode and combine the nodes until nodes are unreachable
	while True:
		print 'ENTERING LOOP'
		print 'APPLYING TO graph'
		distances = distances.flatMap(lambda x: explode_nodes(x, accum, target)) \
													.reduceByKey(reduce_nodes) \
													.filter(lambda (node, info): info[2] != VISITED) \
													.cache()
		
		print 'COUNTING distances'
		# Apply transformations
		distances.count()

		# If accumulator didn't touch more nodes, BFS is complete
		if target in accum.value:
			print 'FOUND TARGET'
			break

	return print_nodes(accum.value, target)




############################################################################
######################################BFS
############################################################################


# Construct path with path_names
def get_path_names(path, page_names):
	path_names = page_names.filter(lambda (id, name): id in path).collect()
	path_with_names = []
	for wiki_id1 in path:
		for (wiki_id2, name) in path_names:
			if wiki_id1 == wiki_id2:
				path_with_names.append(name)
				break
	return path_with_names

# Constructs links as node with edges
def construct_node(line):
	src, dests = line.split(': ')
	dests = [int(to) for to in dests.split(' ')]
	return (int(src), dests)

if __name__ == '__main__':
	sc = SparkContext(appName="P5")
	sc.setLogLevel('WARN')
	print 'GETTING DATA'
	links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
	page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)
	print 'GOT DATA HOE'
	neighbor_graph = links.map(construct_node)
	print 'BUILT THAT GRAPH FAM'

	page_names = page_names.zipWithIndex().map(lambda (n, id): (id+1, n))
	page_names = page_names.sortByKey().cache()
	print 'AYY GOT PAGE NAMES'

	neighbor_graph = neighbor_graph.partitionBy(64).cache()
	Kevin_Bacon = page_names.filter(lambda (k, v): v == 'Kevin_Bacon')
	assert Kevin_Bacon.count() == 1
	Kevin_Bacon = Kevin_Bacon.collect()[0][0]

	Harvard_University = page_names.filter(lambda (k, v): v == 'Harvard_University')
	assert Harvard_University.count() == 1
	Harvard_University = Harvard_University.collect()[0][0]

	print 'GOT IDS STARTING THAT BFS'
	path1 = get_path_names(bfs(sc, neighbor_graph, Kevin_Bacon, Harvard_University), page_names)
	path2 = get_path_names(bfs(sc, neighbor_graph, Harvard_University, Kevin_Bacon), page_names)
	
	print path1
	print path2