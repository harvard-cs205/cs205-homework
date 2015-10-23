from pyspark import SparkContext
sc =SparkContext()

# Create the graph that will be used as input for bfs
# The graph is represented as adjacency data structure as in the original file
def generateGraph(titlefile, linksfile):

	rddTitles = sc.textFile(titlefile)
	rddGraph = sc.textFile(linksfile)

	# Clean the text file my removing ":" and split it to lists
	def replaceChar(x):
		return x.replace(':', '').split(' ')

	# Cast to integer
	def castToInt(x):
		return int(x[0]), [int(i) for i in x[1:]]

	rddGraph = rddGraph.map(replaceChar).map(castToInt)

	# Wiki is 1-indexed.
	rddTitles = rddTitles.zipWithIndex().map(lambda x: (x[1]+1, x[0]))
	return rddGraph.cache(), rddTitles.cache()

# Parallel breadth first search
def bfsParallel(graphRDD, rddTitles, origin, dest, maxDiameter):
	# Convert origin and destination to line numbers.
	start = rddTitles.filter(lambda x: x[1] == origin).collect()[0][0]
	target = rddTitles.filter(lambda x: x[1] == dest).collect()[0][0]

	filterRDD = graphRDD.filter(lambda x: x[0] == start)

	# Set used to record the visited nodes.
	visited = {start}

	# new nodes in a specific level
	newNodes = set()

	# RDD that stores the whole single source distance (Can be discarded in this problem.)
	rdd = sc.parallelize([])

	# level
	i = 0
	stop = False

	# RDD used to store the parent of a node for us to retrieve path later.
	path = sc.parallelize([])
	while not stop:
		rdd += filterRDD.map(lambda x: (x[0],i))

		# Newly visited nodes.
		newNodesRDD = filterRDD.flatMapValues(lambda x: x).values().filter(lambda x: x not in visited)
		newNodes = set(newNodesRDD.collect())
		
		# New visited path's parents
		selectPathRDD = filterRDD.map(lambda x: (x[0], [v for v in x[1] if v not in visited]))
		path += selectPathRDD.map(lambda x: [(v, x[0]) for v in x[1]]).flatMap(lambda x: x)

		# Update visited nodes.
		visited |= newNodes

		# Update total number of nodes visited.
		#numNodesVisited.add(len(newNodes))
		if len(newNodes) == 0: stop = True
		filterRDD = graphRDD.filter(lambda x: x[0] in newNodes)
		i += 1
		if maxDiameter and i == maxDiameter: break

	## Generating the path reversely
	outpath = []
	item = target
	i = 0
	while item != start:
		item = path.filter(lambda x: x[0] == item).collect()[0][1]
		outpath += [item]
		i += 1

	
	# Transform it back to human readable by looking it up in RDDTitles
	setPath = set(outpath)
	readablePath = []
	filteredTitles = rddTitles.filter(lambda x: x[1] in setPath).collectAsMap()
	for item in outpath:
		readablePath = [filteredTitles[item]] + readablePath
	return readablePath

def main():

	# paths to the big data files
	nodes = 'titles-sorted.txt'
	graph = 'links-simple-sorted.txt'

	# setup initial rdds
	rddGraph, rddTitles = generateGraph(nodes, graph)

	start  = 'Kevin_Bacon'
	target = 'Harvard_University'

	outpath  = bfsParallel(rddGraph, rddTitles, start, target, 5)

	# write output to file
	with open('P5.txt', 'wb') as outfile:
		outfile.write('One shortest path generated from ' + start + ' to ' + target + '\n')
		outfile.write('\n')
		outfile.write("->".join(outpath))
		outfile.write('\n')

main()


