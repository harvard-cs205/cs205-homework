import pyspark

sc = pyspark.SparkContext()

# make pyspark shut up
sc.setLogLevel('WARN')

N = 16

def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) for to in dests.split(' ')]
    return (int(src), dests)


links = sc.textFile("s3://Harvard-CS205/wikipedia/links-simple-sorted.txt", N)
page_names = sc.textFile("s3://Harvard-CS205/wikipedia/titles-sorted.txt", N)

page_names = page_names.zipWithIndex().map(lambda (n, id): (id + 1, n))
page_names = page_names.sortByKey().cache()


neighbor_graph = links.map(link_string_to_KV)
neighbor_graph = neighbor_graph.flatMapValues(lambda v: v)
graph = neighbor_graph.partitionBy(N).cache()

opposite_links = graph.map(lambda (u, v): (v, u)).partitionBy(N)
symmetric_graph = graph.union(opposite_links).distinct().partitionBy(N).cache()

intersection = graph.intersection(graph.map(lambda (u, v): (v, u))).partitionBy(N).cache()

def setcontinue((set1, set2)):
	if set1 == set2:
		return (set1, False)
	else:
		if len(set1) > len(set2):
			return (set1, True)
		return (set2, True)


num_components = 0
biggest_component = 0
# get adj list again
graph = graph.groupByKey().mapValues(set).partitionBy(N)
graph = graph.map(lambda (u, v): (u , set([u]) | v), preservesPartitioning=True).cache()
iteration = 0
while not graph.isEmpty():
	print iteration
	allTuples = graph.flatMap(lambda (x,y): [(x,z) for z in y])
	newClusters = graph.join(allTuples.map(lambda (x,y): (y,x)).partitionBy(N), N).mapValues(lambda (s, u): (u, s)).values().reduceByKey(lambda set1, set2: set1 | set2).partitionBy(N)
	graph = graph.join(newClusters, N).mapValues(setcontinue)
	toremove = graph.filter(lambda x: x[1][1] == False)
	clusterstoremove = toremove.map(lambda (x,y): (min(y[0]), y[0])).reduceByKey(lambda x, y: x)
	num_components += clusterstoremove.count()
	maxcomponenthere = len(clusterstoremove.takeOrdered(1, lambda x: -len(x[1]))[0])
	biggest_component = max(biggest_component, maxcomponenthere)
	graph = graph.filter(lambda x: x[1][1] != False).mapValues(lambda (s,b): s).cache()
	iteration += 1

print (num_components, biggest_component)



	





