import pyspark
from pyspark import SparkContext
import time

sc = SparkContext()
sc.setLogLevel('WARN')
links = sc.textFile ('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
page_names = sc.textFile ('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)

def cleanRawLinks(s):
	separate = s.split(":")
	child = separate[1].split()
	return (int(separate[0]), map(int, child))


def cleanRawLinksSymmetrically(s):
	separate = s.split(":")
	child = separate[1].split()
	result = []
	result.append((int(separate[0]), map(int, child)))
	for c in map(int, child):
		result.append((c, [int(separate[0])]))
	return tuple(result)

page_names = page_names.zipWithIndex().map(lambda (n, id): (id + 1, n))
page_names = page_names.sortByKey().cache()
Kevin_Bacon = page_names.filter(lambda (K, V): V == 'Kevin_Bacon').collect()
Kevin_Bacon = Kevin_Bacon[0][0]

Harvard_University = page_names.filter(lambda (K, V): V == 'Harvard_University').collect()
Harvard_University = Harvard_University[0][0]

source = Kevin_Bacon
sink = Harvard_University

cleanLinks = links.map(lambda x: cleanRawLinksSymmetrically(x))
cleanLinks = cleanLinks.map(lambda x: x[0])
cleanLinks = cleanLinks.reduceByKey(lambda x, y: list(set(x + y)))

cleanLinks.cache()


#The below code tests this approach on a small section of the graph
# f = open('ccLog.txt', 'w')
# small_test = sc.textFile('P5_small.txt')
# clean_small_test = small_test.map(lambda x: cleanRawLinksSymmetrically(x))
# clean_small_test = clean_small_test.map(lambda x: x[0])
# clean_small_test = clean_small_test.reduceByKey(lambda x, y: list(set(x + y)))
# clean_small_test.cache()
#[(u'1', u' 1664968'), (u'2', u' 3 747213 1664968 1691047 4095634 5535664')]

def CC(graph, source, sink):
	finished = False
	max_Component_Size = 0
	count = 0
	graph = graph.map(lambda (x, y): (x, (y, 0)) if x == source else (x, (y,  10**9)))
	iteration = 0
	while(not finished):
		graph_and_distances = graph.map(lambda (x, y): (x, (y[0], 0)) if x == source else (x, (y[0],  10**9)))
		stillInCC = True
		while (stillInCC):
			iteration += 1
			print "iteration number: ", iteration
			total_distances_previous = graph_and_distances.values().map(lambda (x, y): y).collect()
			total_d_prev_num = sum(total_distances_previous)
			graph_and_distances = graph_and_distances.flatMap(lambda (x,y): iterate(source, x, y[0], y[1], sink))
			graph_and_distances = graph_and_distances.reduceByKey(lambda y, z: (list(set(y[0] + z[0])), min(y[1], z[1]) ))
			total_distances_after = graph_and_distances.values().map(lambda (x, y): y).collect()
			total_d_after_num = sum(total_distances_after)
			if total_d_after_num == total_d_prev_num:
				count += 1
				print "component found!"
				print "its component number : ", count
				componentSize = graph_and_distances.filter(lambda (x, y): True if y[1] < 10**9 else False).count()
				print "the componentSize is : ", componentSize
				if componentSize > max_Component_Size:
					max_Component_Size = componentSize
				stillInCC = False
				graph = graph_and_distances.filter(lambda (x, y): True if y[1] == 10**9 else False)
				graph_collection = graph.collect()
				if len(graph_collection) == 0:
					finished = True
				else:
					source = graph_collection[0][0]
			print "finished iteration number: ", iteration
	return (count, max_Component_Size)

def iterate(source, nodeName, nodeNeighbors, nodeDepth, sink):
	updates = []
	if nodeDepth < 10**9:
		for neighbor in nodeNeighbors:
			updates.append( (neighbor, ([nodeName], nodeDepth + 1)))
		updatedANode = True
	updates.append((nodeName, (nodeNeighbors, nodeDepth)))
	return updates

answer = CC(cleanLinks, source, sink)
print answer







