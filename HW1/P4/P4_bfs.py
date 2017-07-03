
import pyspark
from pyspark import SparkContext
import time
import sys

#following function does BFS
def BFS(sc, graph, source, maxITER=10):
	#initialize. The nodes have a key (x) which is the character name
	# and then a list of values which include the neighbors, distance (set to 1000 if its not
	# the source at first. 1000 = "unvisited" pretty much -- should pick a much larger number usually
	# but the problem statement said that the graph diameter was 10 sooo...), and a boolean of whether or
	# not this node has been visited before. 
	graph_and_distances = graph.map(lambda (x, y): (x, [y, 0, False]) if x == source else (x, [y,  1000, False]))
	graph_and_distances = graph_and_distances.map(lambda (x, y): (x.encode('utf-8'), [encodeInUTF8(y[0]), y[1], y[2]]))
	Finished = False
	count = 0
	
	while(not Finished):
		#initialize an accumulator. This accumulator is going to serve as telling as whether or not to 
		#terminate based on whether we are still visiting new nodes or if we are done with that. 
		nodesUpdated = sc.accumulator(0)

		#The following commented out code below was an alternative way to tell if my search had terminated
		#in the first part of 4. It checked if the distance was updated in an iteration, and terminated once
		#distances were no longer being updated. Kept it in here for good measure. 
		#The termination code bit that uses this is also commented out a few lines below. 

		# total_distances_previous = graph_and_distances.values().map(lambda (x, y, z): y).collect()
		# total_d_prev_num = sum(total_distances_previous)
		
		#call the method iterate over the arguments here, and pass in the accumulator. 
		#reduce by key to combine all the neighbor nodes we visited by their parent. 
		graph_and_distances = graph_and_distances.flatMap(lambda (x,y): iterate(source, x, y[0], y[1], nodesUpdated, y[2]))
		graph_and_distances = graph_and_distances.reduceByKey(lambda y, z: ( [ list(set(y[0] + z[0])), min(y[1], z[1]), y[2] or z[2]]) )

		#force the calculation of the accumulator. 
		graph_and_distances.count()

		# total_distances_after = graph_and_distances.values().map(lambda (x, y, z): y).collect()
		# total_d_after_num = sum(total_distances_after)

		# if total_d_after_num == total_d_prev_num:
		# 	Finished = True

		#count of how many times we've been through this loop
		count += 1

		#terminate if we aren't visiting any more new nodes.
		if nodesUpdated.value is 0:
			Finished = True

	return graph_and_distances

#encode things in UTF8 because that was giving me trouble earlier. 
def encodeInUTF8(lst):
	for i in xrange(len(lst)):
		lst[i] = lst[i].encode('utf-8')
	return lst

#steps through one iteration of the search. Checks to see if the node we are currently
#processing has been visited so far. If it has, update the distances of its neighbors
#and set their visited to false, and set your own visited to be true. 
#If you update something new, go ahead and increment the accumulator -- you've earned it.
def iterate(source, nodeName, nodeNeighbors, nodeDepth, nUpdate, haveVisitedBefore):
	updates = []
	updatedANode = False
	if nodeDepth <  1000:
		if not haveVisitedBefore:
			for neighbor in nodeNeighbors:
				updates.append( (neighbor, [[nodeName], nodeDepth + 1, False]) )
			nUpdate.add(1)
			updates.append((nodeName, [nodeNeighbors, nodeDepth, True]))
		else:
			updates.append((nodeName, [nodeNeighbors, nodeDepth, True]))	
	else:
		updates.append((nodeName, [nodeNeighbors, nodeDepth, False]))
	return updates





