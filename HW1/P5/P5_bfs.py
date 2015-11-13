from pyspark import AccumulatorParam

class SetAccumulator(AccumulatorParam):
	def zero(self, s):
		return s

	def addInPlace(self, s1, s2):
		s1.update(s2)
		return s1

def parse_link(link):
	this_link, links_to = link.split(': ')
	this_link = int(this_link)
	links_to = map(int, links_to.split())
	return (this_link, links_to)


links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
links = links.map(parse_link)

pages = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')
#  +1 to account for 1 indexing
page_index = pages.zipWithIndex().map(lambda (page, i): (page, i+1))
index_page = page_index.map(lambda (page, i): (i, page))

lookup_by_name = lambda name: page_index.lookup(name)[0]
lookup_by_index = lambda i: index_page.lookup(i)[0]

bacon_index = lookup_by_name('Kevin_Bacon')
harvard_index = lookup_by_name('Harvard_University')

def shortest_path(adjacencies, root, end):
	"""Shortest path from root to end given an adjacency lists; returns None if no path found"""
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
			return map(lookup_by_index, paths.lookup(end)[0])


#  harvard to kevin bacon
print shortest_path(links, harvard_index, bacon_index)

#  kevin bacon to harvard
print shortest_path(links, bacon_index, harvard_index)



