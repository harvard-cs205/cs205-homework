from pyspark import SparkContext
sc = SparkContext()
from P4_bfs import *
from pyspark.accumulators import AccumulatorParam
class VectorAccumulatorParam(AccumulatorParam):
    def zero(self, value):
        return [0.0] * len(value)
    def addInPlace(self, val1, val2):
        for i in xrange(len(val1)):
             val1[i] += val2[i]
        return val1


# Used to create a map from node to link
def chartoVol(s):
	tmp = s.split('"')
	return (tmp[1], [tmp[3]])

# Used to create a map from link to node
def voltoChar(s):
	tmp = s.split('"')
	return (tmp[3], [tmp[1]])

# Generate graph represented as a map where the keys are characters 
# and the values are lists of characters sharing links with the key characters
def createGraph(rdd):
	rddchar = rdd.map(chartoVol)
	rddvol = rdd.map(voltoChar)

	rddChar = rddchar.combineByKey(lambda x: x, lambda x, y: x+y, lambda x,y: x+y)
	dataVol = rddvol.combineByKey(lambda x: x, lambda x, y: x+y, lambda x,y: x+y).collectAsMap()

	def getGraph(x):
		res = set()
		for key in x:
			res |= set(dataVol[key])
		return res

	graph = rddChar.mapValues(getGraph)
	#graph = rddChar.mapValues(getGraph).collectAsMap()
	return graph

nPartitions = 50
data = sc.textFile("source.csv").collect()
rdd = sc.parallelize(data, nPartitions)
graphRDD = createGraph(rdd)

# Run the code for three sample characters.
numNodesVisited = sc.accumulator(1)
dictionary1, log1, totalNumNodesVisited1 = bfsParallel(graphRDD, 'CAPTAIN AMERICA', sc, "", numNodesVisited, False)
numNodesVisited = sc.accumulator(1)
dictionary2, log2, totalNumNodesVisited2 = bfsParallel(graphRDD, 'MISS THING/MARY', sc, "", numNodesVisited, False)
numNodesVisited = sc.accumulator(1)
dictionary3, log3, totalNumNodesVisited3 = bfsParallel(graphRDD, 'ORWELL', sc, "", numNodesVisited, False)

with open("P4.txt", "w") as txtfile:
	txtfile.write('CAPTAIN AMERICA: ')
	txtfile.write("\n")
	txtfile.write(log1)
	txtfile.write("\n")
	txtfile.write("Total nodes visited: " + str(totalNumNodesVisited1))
	txtfile.write("\n")

	txtfile.write('MISS THING/MARY: ')
	txtfile.write("\n")
	txtfile.write(log2)
	txtfile.write("\n")
	txtfile.write("Total nodes visited: " + str(totalNumNodesVisited2))
	txtfile.write("\n")

	txtfile.write('ORWELL: ')
	txtfile.write("\n")
	txtfile.write(log3)
	txtfile.write("\n")
	txtfile.write("Total nodes visited: " + str(totalNumNodesVisited3))
	txtfile.write("\n")

