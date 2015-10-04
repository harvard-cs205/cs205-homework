from pyspark import SparkContext
import itertools
from P4_bfs import *
sc = SparkContext("local", "Simple App")

def addToGraph(characterList):
	"""
	ARGS: characterList is a list of all characters in a single book.
	Add everyone to everyone else's adjacency list
	"""
	for firstCharacter in characterList:
		# If the node hasn't been initialized yet
		if firstCharacter not in adjacencyList:
			adjacencyList[firstCharater] = set()
		# Add every other character to this adjacency list
		for secondCharacter in characterList:
			if firstCharater != secondCharacter:
				adjacencyList[firstCharater].add(secondCharacter)
	return 1

# Details of how this works are in P4.txt
data = sc.textFile("source.csv") \
		 .map(lambda line: (line.split(',')[-1], ','.join(line.split(',')[:-1]))) \
		 .groupByKey() \
		 .flatMap(lambda (book, characters): itertools.permutations(list(characters), 2)) \
		 .groupByKey() \
		 .map(lambda (key, values) : (key, set(values))) \
		 .collect()


# Turn rdd into dict
myAdjacencyList = {}
for item in data:
	myAdjacencyList[item[0]] = item[1]


# print "MISS THING IS " + str(search_bfs_new("MISS THING/MARY", myAdjacencyList, sc))
# print "ORWELL IS " + str(search_bfs_new("ORWELL", myAdjacencyList, sc))
print "CPT AMERICA IS " + str(search_accum("CAPTAIN AMERICA", myAdjacencyList, sc))
