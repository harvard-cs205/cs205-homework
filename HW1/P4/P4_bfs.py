from pyspark import SparkContext

# Function to execute single-source BFS
def ssBFS(graph, sourceNode, sc, diameter=10):
	# Take graph and convert it to the following structure:
	# (char, [neighbors, depth, visited])
	depthGraph = graph.map(lambda (char, neighbs): (char, [neighbs, 0 if char == sourceNode else float("inf"), False]))
	
	# Used to be a fixed loop for diameter size, but accumulator makes 
	# this unnecessary, so making this an inf loop
	while True:
	# for i in range(diameter):
		# This is our queue size. If it ever stays 0 that means we're not
		# adding to the queue anymore and can kick out
		queue = sc.accumulator(0)

		# Execute a single search step and store the new graph 
		# (char, [[neighb1, neighb2, ...], depth, visited])
		stepThrough = depthGraph.flatMap(lambda (char, neighbInfo): executeSearchStep(sourceNode, char, neighbInfo, queue))
		# print stepThrough.take(3)
		# print "------------"

		# Combine touched neighbors of given character through a simple reduction. For every 2 neighbors
		# combine their respective neighbors into a single de-duped set-list, find the minimum depth, 
		# and mark visited as true if at least one was visited. This way we can continue traversing as before.
		# (char, [[neighb1, neighb2, ...], depth, visited])
		combinedNeighbs = stepThrough.reduceByKey(lambda n1, n2: ([list(set(n1[0] + n2[0])), min(n1[1], n2[1]), n1[2] or n2[2]]))
		# print combinedNeighbs.take(3)
		# print "############"

		# Count up the neighbors to update accumulator
		combinedNeighbs.count()

		# See if we accumulated anything this step - of not, BFS is done
		if queue.value is 0:
			return combinedNeighbs

		# Set depthGraph to combined neighbor graph for next iteration
		depthGraph = combinedNeighbs

	# Old return for pre-accumulator BFS
	# return combinedNeighbs

# Function to exectue a single search step. Takes:
# (source, character, [[neighb1, neighb2, ...], depth, visited], accumulator)
# Returns a list of touched nodes in the form:
# [(char, [[neighb1, neighb2, ...], depth, visited]), ...]
def executeSearchStep(sourceNode, char, nextNode, queue):
	neighbs, depth, visited = nextNode
	touchedNodes = []
	# Only expand nodes that have been assigned a depth
	if depth < float("inf"):
		# And those that have not been visited
		if not visited:
			# Build next layer of exploration - make sure links are two-way
			for neighb in neighbs:
				touchedNodes.append((neighb, [[char], depth+1, False]))
			# Increment queue
			queue.add(1)
		# Mark as visited
		touchedNodes.append((char, [neighbs, depth, True]))
	else:
		# Explicitly mark as unvisited
		touchedNodes.append((char, [neighbs, depth, False]))
	return touchedNodes