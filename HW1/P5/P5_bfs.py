from pyspark import SparkContext

# Function to execute single-source BFS with exit node
def ssBFS2(graph, sourceNode, exitNode, sc):
	# Take graph and convert it to the following structure:
	# (char, [neighbors, depth, path])
	depthGraph = graph.map(lambda (char, neighbs): (char, [neighbs, 0 if char == sourceNode else float("inf"), []]))

	# No more accumulator, but we can still use infinite loop 
	# so long as we check when we hit exit node properly
	while True:
		# Execute a single search step and store the new graph 
		# (char, [[neighb1, neighb2, ...], depth, path])
		stepThrough = depthGraph.flatMap(lambda (char, neighbInfo): executeSearchStep(sourceNode, exitNode, char, neighbInfo))
		# print stepThrough.take(3)
		# print "------------"

		# Combine touched neighbors of given character through a simple reduction. For every 2 neighbors
		# combine their respective neighbors into a single de-duped set-list, find the minimum depth, 
		# and concatenate the paths. This way we can continue traversing as before.
		# (char, [[neighb1, neighb2, ...], depth, path])
		combinedNeighbs = stepThrough.reduceByKey(lambda n1, n2: ([list(set(n1[0] + n2[0])), min(n1[1], n2[1]), n1[2] + n2[2]]))
		# print combinedNeighbs.take(3)
		# print "############"
		
		# Lookup the exit node to see if it has been touched (has a dist != inf)
		testReachedExit = combinedNeighbs.lookup(exitNode)[0]
		if testReachedExit[1] < float("inf"):
			return testReachedExit[2]

		# Set depthGraph to combined neighbor graph for next iteration
		depthGraph = combinedNeighbs

# Function to exectue a single search step. Takes:
# (source, exit, character, [[neighb1, neighb2, ...], depth, path])
# Returns a list of touched nodes in the form:
# [(char, [[neighb1, neighb2, ...], depth, path]), ...]
def executeSearchStep(sourceNode, exitNode, char, nextNode):
	neighbs, depth, paths = nextNode
	touchedNodes = []
	# Only expand nodes that have been assigned a depth
	if depth < float("inf"):
		for neighb in neighbs:
			# Update depth and update paths by concatenating char to each one
			# If no paths exist, initiate a new one
			touchedNodes.append((neighb, ([], depth+1, \
								map(lambda path: path + [char], paths) \
								if len(paths) else [[char]])))
	touchedNodes.append((char, (neighbs, depth, [])))
	return touchedNodes