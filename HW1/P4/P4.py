from P4_bfs import *
import pyspark
sc = pyspark.SparkContext()

# Parsing line in text file
def parseLine(line):
	lineSplit = line.split('"')
	comic, char = lineSplit[3], lineSplit[1]
	return (comic, char)

# Create list of character followed by every other character in a particular comic
def permutateChars(x):
	return [(i, list(x[1])) for i in list(x[1])]

# RDD with (character, comic) tuples; swap to produce RDD with (comic, character) tuples
charsInComic = sc.textFile('/Users/idzhang/COLLEGENOW/CS205/source.csv').map(parseLine)

# Group characters featuring in same comic, permutate chars and union all sets of neighbor characters
graph = charsInComic.groupByKey() \
				.flatMap(permutateChars) \
				.mapValues(set) \
				.reduceByKey(lambda s1,s2: s1 | s2)

# Conduct bfs searches on the 3 characters and print final number of visited nodes
chars = ['CAPTAIN AMERICA', 'MISS THING/MARY', 'ORWELL']
for char in chars:
	print len(bfs(graph, char, sc))