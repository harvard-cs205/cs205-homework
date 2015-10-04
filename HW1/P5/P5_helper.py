from pyspark.accumulators import AccumulatorParam
class BFSAccumulatorParam(AccumulatorParam):
	"""
	Defines an accumulator for the optimized search. The 
	accumulator is a dict with key: Character, val: distances
	"""
	def zero(self, value):
		return {}
	def addInPlace(self, val1, val2):
		# A way to merge two dicts and keep the min of conflicts. 
		ret = {}
		for key in list(val1.keys()):
			if key not in val2:
				ret[key] = val1[key]
			else:
				ret[key] = min(val1[key], val2[key], key=lambda (distance, parent): distance)
		for key in list(val2.keys()):
			if key not in val1:
				ret[key] = val2[key]
		return ret

def find_neighbors(adjacencyList, node):
	new_rdd = adjacencyList.filter(lambda k, v: k == node)
	assert len(new_rdd == 1)
	return new_rdd.map(lambda k, v: v).take

def expand_node(frontier_distance):
	def helper((node, (distance, parent, neighbors))):
		if distance == frontier_distance:
			print "Found one"
			return [(new_node, (distance + 1, node, [])) for new_node in neighbors]
		else:
			return [(node, (distance, parent, neighbors))]
	return helper
def combine_nodes((node, distances)):
	ret = min(list(distances), key = lambda (distance, parent, neighbors) : distance)
	neighbors = max(list(distances), key = lambda (distance, parent, neighbors) : len(neighbors))
	return (node, (ret[0], ret[1], neighbors[2]))

def search_bfs_new(startNode, goalNode, adjacencyList, sc):
	make_sure = adjacencyList.filter(lambda (k,v) : k == startNode)
	print make_sure.take(1)
	print "BLAHHH"
	accum = sc.accumulator({}, BFSAccumulatorParam())
	found = 0
	frontier_distance = 0
	while True:
		prev_len = len(accum.value)
		print prev_len
		print "HEYYY"
		def f((node, (distance, parent, neighbors))):
			if distance == frontier_distance:
				accum.add({node:(distance, parent)})
		new_distances = adjacencyList.filter(lambda (node, (distance, parent, neighbors)) : distance == frontier_distance)
		new_distances.foreach(f)
		# If we found the end
		if goalNode in accum.value:
			found = 1
			break
		# If we didn't explore any new nodes
		if len(accum.value) == prev_len:
			break
		# For each node, keep the node in the RDD and add it's neighbors. Take the min distance for ones that have been visited twice
		adjacencyList = adjacencyList.flatMap(expand_node(frontier_distance))
		adjacencyList = adjacencyList.groupByKey().map(combine_nodes)
		# distances = distances.flatMap(lambda (node, (distance, parent, neighbors)): [(node, (distance, parent, neighbors))] + [(newNode, (distance + 1, node)) for newNode in neighbors] if distance == frontier_distance else [(node, (distance, parent))]) \
		# 					 .groupByKey() \
		# 					 .map(lambda (node, distances) : (node, min(list(distances), key = lambda (distance, parent): distance)))
		frontier_distance += 1
	curr_node = goalNode
	ret = [curr_node]
	if found == 1:
		while accum.value[curr_node][1] != None:
			curr_node = accum.value[curr_node][1]
			ret.append(curr_node)
		return ret
	return -1





