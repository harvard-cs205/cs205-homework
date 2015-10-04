from pyspark import SparkContext
import numpy as np

def getIdFromTitle(title):
	return indexByTitle.filter(lambda x:x[1]==title).collect()

def getTitleFromId(nodeId):
	return indexByTitle.filter(lambda x:x[0]==nodeId).collect()

def copartitioned(rdd1,rdd2):
	return rdd1.partitioner == rdd2.partitioner

def bfs(graph,startNodeId,endNodeId):
	gc = graph.count()
	parentGraph = sc.parallelize(zip(np.array(range(gc))+1,[None]*gc)).cache()
	visited = np.array([])
	toVisit = graph.filter(lambda x:x[0]==startNodeId).coalesce(5)
	toVisitCollected = toVisit.collect()
	assert len(toVisitCollected)==1
	notFound = True
	visited = np.append(visited,toVisitCollected[0][0])

	while True:
		print 'visited',visited		
		neighborWithParent = toVisit.flatMap(lambda x:[(y,x[0]) for y in x[1] if y not in visited])
		neighbors = np.unique(np.array(toVisit.flatMap(lambda x:x[1]).collect()))
		#neighbors = np.array(neighborWithParent.keys().distinct().coalesce(100).collect())
		print 'neighbors',neighbors
		print 'neighborWithParent',neighborWithParent.collect()
		notVisitedNeighbors = np.setdiff1d(neighbors,visited)
		if len(notVisitedNeighbors)==0:
			break
		# if len(parentGraph.filter(lambda x:x[0]==startNodeId and x[1]!=None).collect()) >0:
		# 	notFound = False
		# 	break
		parentGraph = parentGraph.union(neighborWithParent).reduceByKey(lambda x,y:x if x!=None else y).cache()
		print 'parentGraph',parentGraph.collect()
		
		# print 'setdiff neighbors-visited',np.setdiff1d(neighbors,visited)
		if endNodeId in notVisitedNeighbors:
			break
		toVisit = graph.filter(lambda x:x[0] in notVisitedNeighbors).coalesce(100).cache()
		visited = np.append(visited,notVisitedNeighbors)
	prevNode = parentGraph.filter(lambda x:x[0]==endNodeId).coalesce(5).collect()
	assert len(prevNode)==1
	while prevNode[0][0] != startNodeId:
		# print '(Child,Parent)',prevNode,'(',getTitleFromId(prevNode[0][0]),',',getTitleFromId(prevNode[0][1]),')'
		print '(Child,Parent)',prevNode
		parent = prevNode[0][1]
		prevNode = parentGraph.filter(lambda x:x[0]==parent).coalesce(5).collect()
		assert len(prevNode)==1



sc=SparkContext()
titlesTxt = sc.textFile('titles-sorted.txt')
txt = sc.textFile('small_links-simple-sorted.txt')
#'s3://Harvard-CS205/wikipedia/links-simple-sorted.txt'
#'s3://Harvard-CS205/wikipedia/titles-sorted.txt'
#txt = sc.textFile('medium-links-simple-sorted.txt')
indexByTitle = titlesTxt.zipWithIndex().map(lambda t: (t[1]+1,t[0])).cache()



graph = txt.map(lambda s:(int(s.split(' ')[0][:-1]), [int(a) for a in s.split(' ')[1:]])).partitionBy(256).cache()


# kevinbaconId = getIdFromTitle('Kevin_Bacon')
# harvard1Id = getIdFromTitle('Harvard_University')
#harvard2Id = getIdFromTitle('Harvard_university')
# assert len(kevinbaconId)==1
# assert len(harvard1Id) == 1
# kevinbaconId = kevinbaconId[0][0]
# harvard1Id = harvard1Id[0][0]
#harvard2Id = harvard2Id[0][0]
#print kevinbaconId,harvard1Id

#print bfs(graph,harvard1Id,kevinbaconId)
print bfs(graph,1,8)












#DO NOT UNCOMMENT
# graph = txt.map(lambda s:(s.split(' ')[0][:-1],(10000, tuple(s.split(' ')[1:]),False)  )  ).map(lambda s:(s[0],(0,s[1][1],True)) if s[0]==startNode else s)
# greyNodes= graph.filter(lambda x:x[0]==startNode)
# for i in range(3):
# 	print 'Grey Count =',greyNodes.count()
# 	neighbors = greyNodes.flatMap(lambda x:x[1][1]).map(lambda x:(x,0))
# 	greyNodes = neighbors.leftOuterJoin(graph).map(lambda x:(x[0],x[1][1]))
	# greyNodes =  graph.filter(f(neighbors))
	# graph = graph.map(extraArgs(neighbors,i))
	# graph.count()
#DO NOT UNCOMMENT


# endNode = '2152852'
# print graph.filter(lambda y:y[0]==endNode).collect()


