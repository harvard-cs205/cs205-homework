from pyspark import SparkContext
import numpy as np
from pyspark.accumulators import AccumulatorParam



sc=SparkContext()

#titlesTxt = sc.textFile('titles-sorted.txt')
#txt = sc.textFile('links-simple-sorted.txt')
txt = sc.textFile('small_links-simple-sorted.txt')
#links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
#indexByTitle = titlesTxt.zipWithIndex().map(lambda t: (t[0],t[1]+1))

# startNode = '2729536'
startNode = '1'
def titleToId(title):
	return str(indexTitle.lookup(title)[0])
graph = txt.map(lambda s:(s.split(' ')[0][:-1],(1000, tuple(s.split(' ')[1:]),False, None))  ).map(lambda s:(s[0],(0, s[1][1], True, None)) if s[0]==startNode else s).filter(lambda x:x[1]!=None).map(lambda x:x).partitionBy(300)

def f(allNeighbors):
	return lambda y:y[0] in allNeighbors

def updateParentAndVisitWrapper(parent,neighbors,distance):
	def updateParentAndVisit(x):
		if  x[1][2] == False and x[0] in neighbors:
			return (x[0],(min(distance,x[1][0]),x[1][1],True,parent))
		else:
			return x
	return updateParentAndVisit

greyNodes= set(graph.filter(lambda x:x[0]==startNode).collect())
visited = np.array([])
for i in range(3):
	print 'numgreynodes=',len(greyNodes)
	allNeighbors = np.array([])
	#toAddToGreyNodes = np.array([])
	for x in greyNodes:
		neighbors = np.array(x[1][1])
		allNeighbors = np.append(allNeighbors, neighbors)
		nodeId = x[0]
		visited = np.append(visited,nodeId)
		graph = graph.map(updateParentAndVisitWrapper(nodeId,neighbors,i+1))
		
		#graph = graph.map(updateParentAndVisitWrapper(nodeId,neighbors,i+1,toAddToGreyNodes))
		#np.append(toAddToGreyNodes,graph.filter(lambda d:d[0] in neighbors).collect())
	greyNodes = set([])
	#greyNodes.update(toAddToGreyNodes)
	allNeighbors = np.setdiff1d(np.unique(allNeighbors),visited)
	broadcastAllNeighbors = sc.broadcast(allNeighbors)
	greyNodes.update(graph.filter(lambda d:d[0] in broadcastAllNeighbors.value).collect())	
	print 'greyNodes',greyNodes
	#greyNodes.update(graph.filter(f(allNeighbors)).collect())
	#print 'allNeighborsLength=',len(allNeighbors)

# endNode = '2152852'
endNode = '3'
prevNode= graph.filter(lambda x:x[0]==endNode).collect()[0]
print 'Last node ',prevNode
while prevNode[1][3]!=None:
	prevNode = graph.filter(lambda x:x[0]==prevNode[1][3]).collect()[0]
	print 'Parent of previous node',prevNode






