import csv
from itertools import permutations
import findspark
findspark.init()
import pyspark
from P4_bfs import bfs1, bfs2

sc = pyspark.SparkContext(appName='marvel')
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

sc.unvisited = sc.accumulator(0)
sc.explored = sc.accumulator(0)

def main():
	adjacencies = load_adjacencies()

	print bfs2(sc, adjacencies, 'CAPTAIN AMERICA')
	print bfs2(sc, adjacencies, 'MISS THING/MARY')
	print bfs2(sc, adjacencies, 'ORWELL')

	from time import sleep
	# sleep(10000)


def load_adjacencies(fname='source.csv'):
	with open('source.csv') as rfile:
		reader = csv.reader(rfile)
		links = [(r[1], r[0]) for r in reader]

	hero_book = sc.parallelize(links, 10)
	create_edges = lambda char_list: tuple(permutations(char_list, 2))
	edges = hero_book.groupByKey().values().flatMap(create_edges)
	adjacencies = edges.groupByKey().map(lambda (n, adj): (n, tuple(set(adj))))
	adjacencies.cache()
	return adjacencies


if __name__ == '__main__':
	main()


