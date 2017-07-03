import pyspark
from pyspark import SparkContext
import time

sc = SparkContext()
sc.setLogLevel('WARN')
#Load in the relevant pages
links = sc.textFile ('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
page_names = sc.textFile ('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)

#clean the links so that you split around the colons
#to get the integers and a map of its integer neighbors.

def cleanRawLinks(s):
	separate = s.split(":")
	child = separate[1].split()
	return (int(separate[0]), map(int, child))

#Get the page names by indexing each id and incrementing by 1 to account
#for the dataset.
page_names = page_names.zipWithIndex().map(lambda (n, id): (id + 1, n))
page_names = page_names.sortByKey().cache()

#Look up and get values for Kevin Bacon and Harvard University
Kevin_Bacon = page_names.filter(lambda (K, V): V == 'Kevin_Bacon').collect()
Kevin_Bacon = Kevin_Bacon[0][0]
Harvard_University = page_names.filter(lambda (K, V): V == 'Harvard_University').collect()
Harvard_University = Harvard_University[0][0]

source = Kevin_Bacon
sink = Harvard_University

cleanLinks = links.map(lambda x: cleanRawLinks(x))
cleanLinks.cache()


# small_test = sc.textFile('P5_small.txt')
# clean_small_test = small_test.map(lambda x: cleanRawLinks(x))
# clean_small_test.cache()


#BFS works similar to last time. Note, I've increased the default distance to 10**9 because I don't
#know anything about the graph right now. Other than that, the major other change is the introduction 
#of a source and a sink and a new stopping condition. 
def BFS(graph, source, sink):
	sinkReached = sc.accumulator(0)
	#initialize like last time, except now we want to keep track of "parents", which is the third
	#list in the values. This is so we can get a path. 
	graph_and_distances = graph.map(lambda (x, y): (x, (y, 0, [])) if x == source else (x, (y,  10**9, [[]])))
	graph_and_distances.collect()
	count = 0
	finished = False
	while(not finished):
		#Below two lines are similar to in P4. 
		graph_and_distances = graph_and_distances.flatMap(lambda (x,y): iterate(source, x, y[0], y[1], y[2], sink))
		graph_and_distances = graph_and_distances.reduceByKey(lambda y, z: (list(set(y[0] + z[0])), min(y[1], z[1]), y[2] + z[2]))
		#You are done once you have explored the Sink node (aka when the key for sink has a distance that is not 10**9)
		done = graph_and_distances.filter(lambda (K, V): True if (K == sink and V[1] < 10**9) else False)
		winner = done.collect()
		#If you are done, then there will be one element in winner. 
		if len(winner) > 0:
			finished = True
			#return the path in winner.
			return winner[0][1][2]
		count +=1
	return graph_and_distances

#Similar to the iterate algorithm in P4 but in this case we don't care about
#the accumulator or visited stopping condition. Rather, we add a path variable
#that allows us to keep track of the parent of this node and the path that 
#gets you to that parent. We set that as the path to this node. 
def iterate(source, nodeName, nodeNeighbors, nodeDepth, prev, sink):
	updates = []
	if nodeDepth < 10**9:
		for neighbor in nodeNeighbors:
			new_paths = []
			if len(prev) > 0:
				for path in prev: 
					 new_paths.append(path + [nodeName])
			else:
				new_paths.append([nodeName])
			updates.append( (neighbor, ([], nodeDepth + 1, new_paths) ))
		updatedANode = True
	updates.append((nodeName, (nodeNeighbors, nodeDepth, [])))
	return updates

#Answer is the result of running the alg on this. Will return all possible "shortest" paths.
answer = BFS(cleanLinks, Kevin_Bacon, Harvard_University)

#The following code gets names of the paths that are in answer
#as these are all ints and thus not as human-readable output. 

named_answers = []
for i in xrange(len(answer3)):
	print i
	path = []
	for k in xrange(len(answer3[i])):
		path.append(page_names.lookup(answer3[i][k]))
	named_answers.append(path)






