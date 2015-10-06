from P4_bfs import *
from pyspark import SparkContext
import itertools

sc = SparkContext(appName="P4")

# Format rows in dataset
def format_line(line):
	row = line.split(',')
	book = row[-1]

	# Join in case character has a comma in it
	character = ','.join(row[:-1])

	return (book, [character])

if __name__ == '__main__':
	# Create rdd of comic book characters
	data = sc.textFile('source.csv')

	# Construct graph
	graph = data.map(format_line) \
					.reduceByKey(lambda x, y: x+y) \
					.flatMap(lambda (book, characters): itertools.permutations(characters, 2)) \
					.map(lambda (node1, node2): (node1, set([node2]))) \
					.reduceByKey(lambda x, y: x | y) \
					.cache()

	# Perform BFS and print length of reachable nodes
	characters = ['"CAPTAIN AMERICA"'] #, '"MISS THING/MARY"', '"ORWELL"']
	for character in characters:
		distances = bfs(graph, character, sc)
		print 'The number of nodes touched when searching for', character, 'is', len(distances)
	raw_input()




