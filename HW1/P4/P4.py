from pyspark import SparkContext
from P4_bfs import ssBFS
from P5_bfs_test import ssBFS2
import itertools
import csv

# Setup file and context
filename = "source.csv"
sc = SparkContext("local", "P4")

# Set up logger so as to be able to see output
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org").setLevel( logger.Level.ERROR )
logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

# Function to find all pair permutations of a list of characters
# Uses Python itertools library
def permutePairs(charList):
	return tuple(itertools.permutations(charList, 2))

# Function to count total touches when running BFS
def countTouches(graph, sourceNode, sc):
	bfsRDD = ssBFS(graph, sourceNode, sc)
	# Filter to return only those elements that have dist < inf, meaning they were touched
	filteredRDD = bfsRDD.filter(lambda (char, neigbs): False if neigbs[1] == float("inf") else True)
	return filteredRDD.count()

# Pre-process the file using Python CSV reader
with open(filename, "rb") as csvfile:
	reader = csv.reader(csvfile)
	revRows = [(book, char) for (char, book) in reader]

# (book, char)
comicRdd = sc.parallelize(revRows)

# Group by book then isolate characters only
# ([char1, char2, ...])
charRdd = comicRdd.groupByKey().values()

# Find all possible character permutations
# ((char1, char2), (char1, char3), ...)
graphUnfilteredEdges = charRdd.flatMap(permutePairs)

# Group the character permutation pairs by first character 
# and filter out duplicates using list set and cache rdd
# ((char1, [char2, char3, ...]), (char2, [char1, char3, ...]), ...)
graph = graphUnfilteredEdges.groupByKey().map(lambda (char, edges): (char, list(set(edges))))
graph = graph.cache()

print "CAPTAIN AMERICA:"
print countTouches(graph, "CAPTAIN AMERICA", sc)
print "MISS THING/MARY:"
print countTouches(graph, "MISS THING/MARY", sc)
print "ORWELL:"
print countTouches(graph, "ORWELL", sc)

print ssBFS2(graph, "MOBIUS", "SERGEI", sc)
