from pyspark import SparkContext
from pyspark.accumulators import AccumulatorParam

class VectorAccumulatorParam(AccumulatorParam):
    def zero(self, value):
        return [0.0] * len(value)
    def addInPlace(self, val1, val2):
        for i in xrange(len(val1)):
             val1[i] += val2[i]
        return val1

# Parallel breadth first search
def bfsParallel(graphRDD, character, sc, log, numNodesVisited, maxDiameter):
	filterRDD = graphRDD.filter(lambda x: x[0] == character)

	# Set used to record the visited nodes.
	visited = {character}

	# new nodes in a specific level
	newNodes = set()

	# list to collect all the distance between a character and its connected characters
	rdd = sc.parallelize([])

	# level
	i = 0
	stop = False
	while not stop:
		rdd += filterRDD.map(lambda x: (x[0], i))
		newNodes = set(filterRDD.flatMapValues(lambda x: x).values().filter(lambda x: x not in visited).collect())

		# Logs to write out the number of nodes outputted to the log.
		log += 'The number of new visited nodes corresponding to depth ' + str(i+1) + ' is ' + str(len(newNodes)) + '\n'

		# Update visited nodes.
		visited |= newNodes

		# Update total number of nodes visited.
		numNodesVisited.add(len(newNodes))
		if len(newNodes) == 0: stop = True
		filterRDD = graphRDD.filter(lambda x: x[0] in newNodes)
		i += 1
		if maxDiameter and i == maxDiameter: break

	return rdd.collectAsMap(), log, numNodesVisited.value










