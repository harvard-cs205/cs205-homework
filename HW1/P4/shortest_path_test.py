from pyspark import AccumulatorParam

class SetAccumulator(AccumulatorParam):
	def zero(self, s):
		return s

	def addInPlace(self, s1, s2):
		s1.update(s2)
		return s1

def shortest_path(sc, adjacencies, root, end):
	"""Test of P5 shortest path problem"""

	paths = sc.parallelize([(root, [root])])
	queue = set([root])
	visited = set([root])

	while (True):
		accumulated_to_visit = sc.accumulator(set([]), SetAccumulator())
		to_visit = adjacencies.filter(lambda (n, adj): n in queue)
		to_visit.foreach(lambda (n, adj): accumulated_to_visit.add(set(adj)))

		paths = to_visit.join(paths).flatMap(lambda (n, (adj, prev_path)): [(a, prev_path + [a]) for a in adj])
		paths = paths.reduceByKey(lambda x,y: x) #  take an arbitrary path if there are multiple

		queue = accumulated_to_visit.value - visited
		if not len(queue):
			print 'Processed whole connected component without finding %s' % end
			return None
		visited.update(accumulated_to_visit.value)
		if end in visited:
			return paths.lookup(end)
