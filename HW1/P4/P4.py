import csv
from itertools import permutations
import findspark
findspark.init()
import pyspark
from P4_bfs import bfs1, bfs2
from shortest_path_test import shortest_path

sc = pyspark.SparkContext(appName='marvel')
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

sc.unvisited = sc.accumulator(0)
sc.explored = sc.accumulator(0)

def main():
	adjacencies = load_adjacencies(cache=False)

	characters = ['CAPTAIN AMERICA', 'MISS THING/MARY', 'ORWELL']
	for c in characters:
		print 'BFS starting from %s touches %d nodes' % (c, bfs2(sc, adjacencies, c))


def load_adjacencies(fname='source.csv', cache=False):
	with open('source.csv') as rfile:
		reader = csv.reader(rfile)
		links = [(r[1], r[0]) for r in reader]

	hero_book = sc.parallelize(links)
	create_edges = lambda char_list: tuple(permutations(char_list, 2))
	edges = hero_book.groupByKey().values().flatMap(create_edges)
	adjacencies = edges.groupByKey().map(lambda (n, adj): (n, tuple(set(adj))))
	if cache:
		adjacencies.cache()
	return adjacencies


if __name__ == '__main__':
	main()


