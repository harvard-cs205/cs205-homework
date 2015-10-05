# Wrapper code that loads the data, creates the graph RDD, and uses the functions you write in P4 bfs.py to search for the 
# given nodes above.

from P4_bfs import *
import pyspark
sc = pyspark.SparkContext()
accum = sc.accumulator(0)

def parseLine(line):
	lineSplit = line.split('"')
	comic, char = lineSplit[3], lineSplit[1]
	return (comic, char)

def permutateChars(x):
	return [(i, list(x[1])) for i in list(x[1])]

# RDD with (character, comic) tuples; swap to produce RDD with (comic, character) tuples
charsInComic = sc.textFile('/Users/idzhang/COLLEGENOW/CS205/source.csv').map(parseLine)

# Group characters featuring in same comic and permutate chars
graph = charsInComic.groupByKey() \
				.flatMap(permutateChars) \
				.mapValues(set) \
				.reduceByKey(lambda s1,s2: s1 | s2)

# Conduct bfs searches and log number of visited nodes after max 10 iterations
print len(bfs(graph, 'CAPTAIN AMERICA'))
print len(bfs(graph, 'MISS THING/MARY'))
print len(bfs(graph, 'ORWELL'))