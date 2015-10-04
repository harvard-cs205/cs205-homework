from pyspark import SparkContext
import numpy as np
import time

def getIdFromTitle(title):
	return indexByTitle.filter(lambda x:x[1]==title).collect()

def getTitleFromId(nodeId):
	return indexByTitle.filter(lambda x:x[0]==nodeId).collect()

def copartitioned(rdd1,rdd2):
	return rdd1.partitioner == rdd2.partitioner

def bfs(graph,startNodeId,endNodeId):
	gc = graph.count()
	#Parent graph. Format: (Child,Parent) initialized to (NodeId,None) since nodes don't have parents to start with
	parentGraph = sc.parallelize(zip(np.array(range(gc))+1,[None]*gc)).cache()
	#Put visited node IDs in here
	visited = np.array([])
	#Initialize toVisit with the starting node
	toVisit = graph.filter(lambda x:x[0]==startNodeId).coalesce(5)
	toVisitCollected = toVisit.collect()
	#Make sure we only get back one node
	assert len(toVisitCollected)==1
	#Starting node marked as visited
	visited = np.append(visited,toVisitCollected[0][0])
	while True:
		#Convert (nodeId,[neighbors]) to [(neighbor1,nodeId),(neighbor2,nodeId)...]
		neighborWithParent = toVisit.flatMap(lambda x:[(y,x[0]) for y in x[1] if y not in visited])
		#Collect the neighbors(we will visit them next)
		neighbors = np.unique(np.array(toVisit.flatMap(lambda x:x[1]).collect()))
		#There will be duplicates, so subtract those that have already been visited
		notVisitedNeighbors = np.setdiff1d(neighbors,visited)
		#Stop traversing graph if no more new neighbors to visit
		if len(notVisitedNeighbors)==0:
			break
		#Update parent graph. We take our (child,parent) pairs and union them with our parent graph and reduce by key to replace (NodeId,None) with (Child,Parent) 
		parentGraph = parentGraph.union(neighborWithParent).reduceByKey(lambda x,y:x if x!=None else y).cache()
		#If end node is in the neighbors to visit, no need to visit, just break out of loop
		if endNodeId in notVisitedNeighbors:
			break
		#Retrieve those nodes to visit from graph
		toVisit = graph.filter(lambda x:x[0] in notVisitedNeighbors).coalesce(100).cache()
		#Mark them as visited
		visited = np.append(visited,notVisitedNeighbors)

	#Print the chain of nodes
	prevNode = parentGraph.filter(lambda x:x[0]==endNodeId).coalesce(5).collect()
	assert len(prevNode)==1
	while prevNode[0][0] != startNodeId:
		print '(Child,Parent)',prevNode,'(',getTitleFromId(prevNode[0][0]),',',getTitleFromId(prevNode[0][1]),')'
		print '(Child,Parent)',prevNode
		parent = prevNode[0][1]
		prevNode = parentGraph.filter(lambda x:x[0]==parent).coalesce(5).collect()
		assert len(prevNode)==1

sc=SparkContext()
titlesTxt = sc.textFile('titles-sorted.txt')
txt = sc.textFile('links-simple-sorted.txt')
#Convert titles dataset to RDD of format: (index,title)
indexByTitle = titlesTxt.zipWithIndex().map(lambda t: (t[1]+1,t[0])).cache()
#Initialize graph with format: (nodeId,[neighborId1,neighborId2...])
graph = txt.map(lambda s:(int(s.split(' ')[0][:-1]), [int(a) for a in s.split(' ')[1:]])).partitionBy(256).cache()


kevinbaconId = getIdFromTitle('Kevin_Bacon')
harvard1Id = getIdFromTitle('Harvard_University')
#Sanity check
assert len(kevinbaconId)==1
assert len(harvard1Id) == 1
kevinbaconId = kevinbaconId[0][0]
harvard1Id = harvard1Id[0][0]

startTime = time.time()
print bfs(graph,kevinbaconId,harvard1Id)
print 'end time=',time.time()-startTime

#'s3://Harvard-CS205/wikipedia/links-simple-sorted.txt'
#'s3://Harvard-CS205/wikipedia/titles-sorted.txt'
#txt = sc.textFile('medium-links-simple-sorted.txt')


