#######
# Implements Breadth First Search
# Arguments:
# 	adj (KV RDD): Edge list. For example: [(1,2), (1, 3), (2,3), (3, 2)] is the graph where there is a directed edge from 1 to 2 and 1 to 3, edges in both directions between 2 and 3.
# 	start (string): Where the breadth first search will start
# Returns:
# 	(RDD) Distances to each point, and (RDD) list of all points that were not reachable.
def bfs(adj, start, sc):
	accum = sc.accumulator(0)

	def reduceFun(tup):
		#global accum
		if tup[0] == None:
			#accum.add(1)
			return tup[1]
		elif tup[1] == None:
			return tup[0]
		else:
			return min(tup[0], tup[1])

	def reduceFun2(tup):
		if tup[1] == None:
			return tup[0]
		else:
			return max(tup[0],tup[1])



	distances = adj.map(lambda (node, neighbor): (node, -1.0)).distinct()
	traversed = sc.parallelize([(start, 0)])
	farthest = 0
	accum.add(1)
	while accum.value != 0:
		accum.value = 0
		print "\n\nOn iteration ", farthest, ' for ', start, '\n\n'
		# get all of the neighboring superheros
		#	Start by filtering for the ones that are farthest away
		# 	Then join this with the adjacency matrix, which gives you all of the places it can go
		#	Get just the neighbors by using .values()
		# 	Get rid of the distance value from the left side
		#	Keep only the unique ones using distinct (since there could be multipe ways to reach a node)
		#	Add the distance to them, using a lambda that makes the neighbors into KVs with Key = name, Value = dist
		neighbors = traversed.filter(lambda (node, dist): dist == farthest).join(adj).values().map(lambda x: x[1]).distinct().map(lambda x: (x, farthest + 1))
		# combine the neighbors with what we already had, removing values we have already seen.
		traversed = traversed.fullOuterJoin(neighbors).mapValues(reduceFun).distinct()
		#traversed.count()
		# force the eval to calculate the accum values.
		farthest += 1
		traversed.filter(lambda (x, dist): dist == farthest).foreach(lambda x: accum.add(1))

	final_vals = distances.leftOuterJoin(traversed).mapValues(reduceFun2)
	#print "Distance Distribution for ", start 
	#print final_vals.values().countByValue(), '\n'
	return final_vals, final_vals.filter(lambda (node, dist): dist < 0)