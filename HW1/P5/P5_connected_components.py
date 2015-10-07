from __future__ import division
import numpy as np
import pyspark

sc = pyspark.SparkContext(appName = "Spark1")

# code below adopted from Ray Jones Github
# -------------------
def getAdjList(rdd):
	src,dests = rdd.split(': ')
	dests = [int(to) for to in dests.split(' ')]
	return (int(src),dests)

links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt',32)
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt',32)

adjList = links.map(getAdjList)
adjList = adjList.partitionBy(70).cache()

page_names = page_names.zipWithIndex().map(lambda (n,id): (id+1,n)) # (uuid,name)
page_names = page_names.sortByKey().cache()
# -------------------
# The code below forms the adjacency matrices for the two cases.
# The first is the case when each edge is taken to be symmetric
# The second is the case when each edge exists only if it is symmetric to begin with

#undirAdjList = adjList.map(lambda x: [(x[0],ii) for ii in x[1]]).flatMap(lambda x: x)
#undirAdjList = undirAdjList.map(lambda x: [x,(x[1],x[0])]).flatMap(lambda x: x)
#undirAdjList = undirAdjList.groupByKey().mapValues(list).map(lambda x: (x[0],np.unique(x[1]))).partitionBy(70)

symmAdjList = adjList.map(lambda x: [(x[0],ii) for ii in x[1]]).flatMap(lambda x: x)
symmAdjList = symmAdjList.map(lambda x: [x,(x[1],x[0])]).flatMap(lambda x: x)
symmAdjList = symmAdjList.map(lambda x: np.sort(x)).map(lambda x: (x[0],x[1]))
symmAdjList = symmAdjList.map(lambda x: (x,1)).reduceByKey(lambda x,y: x+y)
symmAdjList = symmAdjList.filter(lambda x: x[1] == 2).map(lambda x: [x[0],(x[0][1],x[0][0])]).flatMap(lambda x: x)
symmAdjList = symmAdjList.groupByKey().mapValues(list).map(lambda x: (x[0],np.unique(x[1]))).partitionBy(70)

characterList = page_names.map(lambda x: x[0]) # uuid
nameIDs = characterList.zipWithIndex() # (uuid, index)
startNames = nameIDs.map(lambda (n,id): (id,n)) # (index, uuid)
startNames = startNames.sortByKey().partitionBy(70).cache()
componentSizes = []

numIt = 0

while startNames.count() > 0: # continue until all nodes have been explored
	bfsDistances = characterList.map(lambda x: (x,float('inf'))).partitionBy(70).cache() # (uuid,inf)
	
	nNamesLeft = startNames.count()
	startName = startNames.take(min([nNamesLeft,500])) # take 500 nodes to start BFS
	currentSet = sc.parallelize([ii[1] for ii in startName]) # uuid
	currentSet = currentSet.map(lambda x: (x,x)).partitionBy(70).cache() # start with (uuid, membership (= uuid initially))
	counts = []

	nNeighbors = sc.accumulator(0)

	def updateDist(node):
		if node[1][1] != None: # value present
			if node[1][0] > node[1][1]:
				nNeighbors.add(1)
			return (node[0],min([node[1][0],node[1][1]])) # (uuid, membership), update membership if neighbor has lower membership
		else:
			return (node[0],node[1][0])
	count = 0
	oldValue = 0
	newValue = 1

	while newValue != oldValue:
		bfsDistances = bfsDistances.leftOuterJoin(currentSet).cache() # (character,(storedMembership,neighborMembership)), if currentSetDist = none, do nothing 
		bfsDistances.count()	
		oldValue = nNeighbors.value                                                      
		bfsDistances = bfsDistances.map(updateDist).cache() # update distances (character,updatedMembership)
		bfsDistances.count()
		newValue = nNeighbors.value
		if newValue != oldValue:
			counts.append(newValue)
			currentSet = currentSet.rightOuterJoin(symmAdjList)
			currentSet = currentSet.filter(lambda x: x[1][0] != None) # for nodes in current set, get (node, (membership,[node.neighbors]))
			currentSet = currentSet.map(lambda x: (x[1][0],x[1][1])) # (membership,[node.neighbors])
			currentSet = currentSet.map(lambda x: [(ii,x[0]) for ii in x[1]]) #[(neighbor1,membership),(neighbor2,membership),...],...
			currentSet = currentSet.flatMap(lambda x: x).groupByKey().mapValues(list).map(lambda x: (x[0],min(x[1]))) # (neighbor1,min(previousMem)]),(neighbor2,min(previousMem)),...
			bfsDistances = bfsDistances.partitionBy(70).cache()
			currentSet = currentSet.partitionBy(70).cache()
			count += 1

	bfsDistances = bfsDistances.partitionBy(70).cache()
	
	touchedNodes = bfsDistances.filter(lambda x: x[1] != float('inf'))
	touchedNodesC = touchedNodes.map(lambda x: (x[1],1)).reduceByKey(lambda x,y: x+y).map(lambda x: x[1])
	componentSizes = componentSizes + touchedNodesC.collect()
	touchedNodes = touchedNodes.join(nameIDs).map(lambda x: (x[0],x[1][1])).map(lambda x: (x[1],x[0])) # (index,uuid)
	startNames = startNames.subtract(touchedNodes).partitionBy(70).cache()
	numIt = numIt + 1

print componentSizes
print len(componentSizes)
print max(componentSizes)



