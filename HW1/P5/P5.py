import pyspark
from pyspark import AccumulatorParam
# sc = pyspark.SparkContext()
sc.setLogLevel('WARN')

### Helper Functions ###
# Parse links string into key-value pair
def parseString(s):
	src, dests = s.split(': ')
	dests = [int(to) for to in dests.split(' ')]
	return (int(src), dests)

# Combine two dictionaries in accumulator, giving v1 priority if keys collide
def combineDicts(v1,v2):
	combine = v2.copy()
	combine.update(v1)
	return combine

# Returns dictionary mapping all children to its immediate parent in shortest path from source
def childrenToParent(parent, children):
	d = {}
	for child in children:
		d[child] = parent
	return d

# Set custom accumulator class for dictionaries
class AccumulatorParamDict(AccumulatorParam):
	def zero(self, initialValue):
		return {}
	def addInPlace(self, v1, v2):
		return combineDicts(v1,v2)

# Parallelized breadth first search
def bfs(start, end, links, sc):
	# Queue keeps track of nodes to be explored at each level of BFS
	q = set([start])
	# Accumulator has keys that keeps track of all visited nodes, and values being node's immediate
	# parent on shortest path from start
	visited = sc.accumulator({start: None}, AccumulatorParamDict())
	# Distance from start node
	dist = 0
	while q:
		print 'STARTING BFS iteration ' + str(dist)
		# Keeps track of nodes that have already been visited before current iteration
		prevVisited = set(list(visited.value.keys()))
		# Finds new nodes to explore in this iteration, and keep track of parent first time node is
		# explored in BFS
		links.filter(lambda (k,v): k in q).foreach(lambda (k,v): visited.add(childrenToParent(k,v)))
		# New nodes that need to be explored in next iteration
		q = set(visited.value.keys()) - prevVisited
		dist += 1
		# If end is found, break out of BFS
		if end in visited.value.keys():
			break
	# Find path by following parents from dest until start is found
	path = [end]
	while end != start:
		end = visited.value[end]
		path.append(end)	
	# Reverse list to find path from start to end
	path = path[::-1]
	return (path, dist)

# Read in files from s3 before parsing and preprocessing
links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
links = links.map(parseString).partitionBy(64).cache()

# Find indices for Harvard_University and Kevin_Bacon
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')
page_names = page_names.zipWithIndex().map(lambda (k,v): (k,v+1))
harvard = page_names.lookup("Harvard_University")[0]
bacon = page_names.lookup("Kevin_Bacon")[0]

# Use pages rdd to lookup page name from node id
pages = page_names.map(lambda (k,v): (v,k)).cache()

# Conduct bfs from Harvard_University to Kevin_Bacon and the reverse
(path, dist) = bfs(bacon, harvard, links, sc)
(path2, dist2) = bfs(harvard, bacon, links, sc)

# Print results
print 'The shortest path from Kevin_Bacon to Harvard_University has distance %s; path is: ' % str(dist)
for node in path:
	print pages.lookup(node)

print 'The shortest path from Harvard_University and Kevin_Bacon has distance %s; path is: ' % str(dist2)
for node in path2:
	print pages.lookup(node)
