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
		# Code from http://stackoverflow.com/questions/28642671/merge-dictionaries-with-minimum-value-of-common-keys
		return {key: min(i for i in (val1.get(key), val2.get(key)) if i >= 0) for key in val1.viewkeys() | val2}

def search_accum(startNode, adjacencyList, sc):
	"""
	The Goal of this function was to eliminate shuffle costs by 
	using a broadcast variable. It is slightly buggy so I just went the 
	regular way with search_bfs_new
	"""
	# Init the accumulator
	accum = sc.accumulator({}, BFSAccumulatorParam())
	# The rdd with Key: character, Val: Distances
	distances = sc.parallelize([(startNode, 0)])
	# A variable to keep track of nodes we have visited
	seen = sc.broadcast(set())

	# While we still have unvisited nodes
	while not distances.isEmpty():
		# A helper function to add ot the accumulator
		def f(listvalue):
			accum.add({listvalue[0]:listvalue[1]})

		distances.foreach(f)
		# Add all the elements in the rdd to to the seen set
		current = distances.collect()
		seen = sc.broadcast(seen.value.union(set([node[0] for node in current])))
		# Remove element from RDD and add neighbors if we haven't seen them
		distances = distances.flatMap(lambda (node, distance): [(newNode, distance + 1) for newNode in adjacencyList[node] if newNode not in seen.value])
	# Could also return the accumulator itself
	return len(accum.value)


def search_bfs(startNode, adjacencyList, sc):
	distances = sc.parallelize([(startNode, 0)])
	for i in range(10):
		# For each node, keep the node in the RDD and add it's neighbors. Take the min distance for ones that have been visited twice
		distances = distances.flatMap(lambda (node, distance): [(node, distance)] + [(newNode, distance + 1) for newNode in adjacencyList[node]] if distance == i else [(node, distance)]) \
							 .groupByKey() \
							 .map(lambda (node, distances) : (node, min(list(distances))))
		print len(distances.collect())
	# Could also return distances.collect()
	return distances.count()

def search_bfs_new(startNode, adjacencyList, sc):
	"""
	Uses an accumulator to find the path
	"""
	distances = sc.parallelize([(startNode, 0)])
	accum = sc.accumulator({}, BFSAccumulatorParam())

	frontier_distance = 0
	while True:
		prev_len = len(accum.value)
		def f(listvalue):
			accum.add({listvalue[0]:listvalue[1]})

		new_distances = distances.filter(lambda (node, distance) : distance == frontier_distance)
		new_distances.foreach(f)

		if len(accum.value) == prev_len:
			break


		# For each node, keep the node in the RDD and add it's neighbors. Take the min distance for ones that have been visited twice
		distances = distances.flatMap(lambda (node, distance): [(node, distance)] + [(newNode, distance + 1) for newNode in adjacencyList[node]] if distance == frontier_distance else [(node, distance)]) \
							 .groupByKey() \
							 .map(lambda (node, distances) : (node, min(list(distances))))
		
		frontier_distance += 1

		#print len(distances.collect())
	# Could also return distances.collect()
	return len(accum.value)


