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
adjList = adjList.partitionBy(30).cache()

page_names = page_names.zipWithIndex().map(lambda (n,id): (id+1,n))
page_names = page_names.sortByKey().cache()
# -------------------

bfsDistances = page_names.map(lambda x: (x[0],(None,float('inf')))).partitionBy(30).cache()
#bfsDistances.count()

count = 0
startName = 'Kevin_Bacon'
startID = page_names.filter(lambda (K,V): V == startName).collect()[0][0]
currentSet = sc.parallelize([startID])
currentSet = currentSet.map(lambda x: (x,(None,float('inf')))).partitionBy(30).cache() # start with (charName, distance)
counts = []

targetName = 'Harvard_University'
targetID = page_names.filter(lambda (K,V): V == targetName).collect()[0][0]

nNeighbors = sc.accumulator(0)

def updateDist(node):
	if node[1][1] != None: # value present
		if node[1][1][1] == float('inf'):
			nNeighbors.add(1)
		return (node[0],(node[1][1][0],min([node[1][0][1],count])))
	else:
		return (node[0],node[1][0])

oldValue = 0
newValue = 1

while newValue != oldValue:
	bfsDistances = bfsDistances.leftOuterJoin(currentSet).cache() # (character,((storedPrevious,storedDist),(setDistPrevious,currentSetDist))), if currentSetDist = none, do nothing 
	bfsDistances.count()                                                         						#    else if new neighbor ('inf'), replace with count
	oldValue = nNeighbors.value
	bfsDistances = bfsDistances.map(updateDist).cache() # update distances (character,(prev,updatedDistance))
	bfsDistances.count()
	newValue = nNeighbors.value
	currentTravel = bfsDistances.filter(lambda x: x[0] == targetID).collect()
	currentTravel = currentTravel[0][1][1]
	if  currentTravel != float('inf'):
		print currentTravel
		break
	if newValue != oldValue:
		counts.append(newValue)
		currentSet = currentSet.rightOuterJoin(adjList)
		currentSet = currentSet.filter(lambda x: x[1][0] != None) # for nodes in current set, get (node, ((None,inf),[node.neighbors]))
		currentSet = currentSet.map(lambda x: (x[0],x[1][1])) # (node,[node.neighbors])
		currentSet = currentSet.map(lambda x: [(ii,x[0]) for ii in x[1]]) #[(neighbor1,node),(neighbor2,node),...]
		currentSet = currentSet.flatMap(lambda x: x).groupByKey().mapValues(list).map(lambda x: (x[0],np.unique(x[1]))) # (neighbor1,[previousNodes1]),(neighbor2,[previousNodes]),...
		currentSet = currentSet.join(bfsDistances).map(lambda x: (x[0],(x[1][0],x[1][1][1]))) # map current distance (neighbor,([previousNodes1],distance))
		currentSet = currentSet.filter(lambda x: x[1][1] == float('inf')) # check if neighbor has aleady been explored, currentSet should all be (neighbor,([previousNodes1],inf))
		bfsDistances = bfsDistances.partitionBy(30).cache()
		currentSet = currentSet.partitionBy(30).cache()
		count += 1

print currentTravel
print oldValue
print newValue
print counts


bfsDistances = bfsDistances.partitionBy(30).cache() # (character,(prevNodes,updatedDistance))
endNodes = sc.parallelize([targetID]).map(lambda x: (x,None)).partitionBy(30).cache()

def getPaths(endNodes): # endNodes = (currentNodes1, prevNodes1),(currentNodes1, prevNodes2),...
	rev = endNodes.map(lambda x: (x[1],x[0])) # (prevNodes1, currentNodes1),(prevNodes2, currentNodes1)
	firstValue = rev.map(lambda x: x[0]).distinct().collect()[0]
	if firstValue == startID:
		endNodes = endNodes.map(lambda x: (x[0],[x[1]])).partitionBy(30).cache()
		return endNodes
	else:
		if endNodes.map(lambda x: x[1]).collect()[0] == None:
			endNodes = endNodes.join(bfsDistances).map(lambda x: (x[0],x[1][1][0])).partitionBy(30).cache() # (targetNode, previousNodes),...
			endNodes = endNodes.map(lambda x: [(x[0],ii) for ii in x[1]]) # [(targetNode, previousNode1), (targetNode, previousNode2),...]
			endNodes = endNodes.flatMap(lambda x: x).partitionBy(30).cache() # (targetNode, previousNode1), (targetNode, previousNode2),...
			return getPaths(endNodes)
		else:
			intermediate = endNodes.map(lambda x: (x[1],0)) # (prevNode1,0),(prevNode2,0),...
			intermediate = intermediate.join(bfsDistances).map(lambda x: (x[0],x[1][1][0])) # (prevNode1, prevPrevNodes1),...
			#intermediate = intermediate.map(lambda x: [(x[0],ii) for ii in x[1]] if x[1][0] != startName else [(x[0],x[1][0])]) # [(prevNode1, prevPrevNode1), (prevNode1, prevPrevNode2),...]
			intermediate = intermediate.map(lambda x: [(x[0],ii) for ii in x[1]]) # [(prevNode1, prevPrevNode1), (prevNode1, prevPrevNode2),...]
			intermediate = intermediate.flatMap(lambda x: x).partitionBy(30).cache() # (prevNode1, prevPrevNode1), (prevNode1, prevPrevNode2),...
			
			nextIt = getPaths(intermediate) # (prevNode1,prevPrevNodes1),(prevNode1,prevPrevNodes2)
			revEndNodes = endNodes.map(lambda x: (x[1],x[0])) # (prevNode1,currentNodes1),(prevNode2,currentNodes1),...
			appendPath = revEndNodes.join(nextIt) # (prevNode1, (currentNodes1,prevPrevNodes1))
			appendPath = appendPath.map(lambda x: (x[1][0],[x[0]]+x[1][1])).partitionBy(30).cache() # (currentNodes1,prevNodesUpdated)
			return appendPath

testPaths = getPaths(endNodes)
testPaths = testPaths.map(lambda x: [x[0]]+x[1])
testPaths = testPaths.map(lambda x: [x[len(x)-1-ii] for ii in range(0,len(x))])
print testPaths.collect()





