import csv
from itertools import permutations
import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext(appName='marvel')
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )



def main():
	adjacencies = load_adjacencies()

	print bfs(adjacencies, 'CAPTAIN AMERICA')



def load_adjacencies(fname='source.csv'):
	with open('source.csv') as rfile:
		reader = csv.reader(rfile)
		links = [(r[1], r[0]) for r in reader]

	hero_book = sc.parallelize(links)
	create_edges = lambda char_list: tuple(permutations(char_list, 2))
	edges = hero_book.groupByKey().values().flatMap(create_edges)
	adjacencies = edges.groupByKey().map(lambda (n, adj): (n, tuple(set(adj))))
	adjacencies.cache()
	return adjacencies



#outer join with adjacencies and filter out non-nulls, then map to remove the extra column while counting whats left unexplored

# visited_this_iteration = adjacencies.join(to_visit)
# adjacencies = adjacencies.leftOuterJoin(visited_this_iteration)
unvisited = sc.accumulator(0)
explored = sc.accumulator(0)

def to_be_visited((n, (adj, distance))):
	if distance is not None:
		explored.add(1)
		return True
	unvisited.add(1)
	return False
# lambda (n, (adj, distance)): distance is not None

def bfs(adjacencies, root):
	to_visit = sc.parallelize([(root, 0)])
	distances = sc.parallelize([])
	last_visited_state = (-1, -1)


	for i in range(100):
		last_unvisited = unvisited.value
		last_explored = explored.value

		visited_this_iteration = adjacencies.leftOuterJoin(to_visit).filter(to_be_visited)
		visited_this_iteration.count() # force evaluation so accumulators update
		distances = visited_this_iteration.map(lambda (n, (adj, distance)): (n, distance)).union(distances).reduceByKey(min)
		to_visit = visited_this_iteration.flatMap(lambda (n, (adj, distance)): adj).distinct().map(lambda a:(a, i))

		visited_state = (unvisited.value - last_unvisited, explored.value - last_explored)
		if visited_state == last_visited_state:
			break
		else:
			last_visited_state = visited_state

	return distances.count()

def inefficient_bfs(adjacencies, root, graph_diameter=10):
	"""BFS on adjacency matrix from root, returns number of nodes explored"""

	inf = float('inf')
	set_initial_distance = lambda (n, adj) : (n, 0 if n == root else inf)
	distances = adjacencies.map(set_initial_distance)

	queue = set([root])
	for i in range(1, graph_diameter+1):
		to_process = lambda (n, adj): n in queue
		to_add = lambda (n, adj): adj
		queue = set(adjacencies.filter(to_process).flatMap(to_add).collect())
		distances = distances.map(lambda (n, d): (n, min(d, i)) if n in queue else (n, d))
		distances = sc.parallelize(distances.collect())

	explored = lambda (n, distance): distance < inf
	n_nodes_touched = len(distances.filter(explored).collect())

	return n_nodes_touched

	# for i in range(10):
	# 	explored = lambda (n, distance): distance == i
	# 	d = distances.filter(explored).collect()
	# 	count = len(distances.collect())
	# 	print i, len(d), count


if __name__ == '__main__':
	main()


