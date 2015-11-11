from __future__ import division
import numpy as np
import pyspark

def bfsSearch(startName, charAdj, sc):

	currentSet = sc.parallelize([startName])
	currentSet = currentSet.map(lambda x: (x,(None,float('inf')))).partitionBy(10).cache()
	characterList = charAdj.map(lambda x: x[0])
	counts = []
	count = 0

	bfsDistances = characterList.map(lambda x: (x,(None,float('inf')))).partitionBy(10).cache()

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
		bfsDistances.count() 													#       else if new neighbor ('inf'), replace with count
		oldValue = nNeighbors.value
		bfsDistances = bfsDistances.map(updateDist).cache() # update distances (character,(prevNodes,updatedDistance))
		bfsDistances.count()
		newValue = nNeighbors.value
		if newValue != oldValue:
			counts.append(newValue)
			currentSet = currentSet.rightOuterJoin(charAdj)
			currentSet = currentSet.filter(lambda x: x[1][0] != None) # for nodes in current set, get (node, ((None,inf),[node.neighbors]))
			currentSet = currentSet.map(lambda x: (x[0],x[1][1])) # (node,[node.neighbors])
			currentSet = currentSet.map(lambda x: [(ii,x[0]) for ii in x[1]]) #[(neighbor1,node),(neighbor2,node),...]
			currentSet = currentSet.flatMap(lambda x: x).groupByKey().mapValues(list).map(lambda x: (x[0],np.unique(x[1]))) # (neighbor1,[previousNodes1]),(neighbor2,[previousNodes2]),...
			currentSet = currentSet.join(bfsDistances).map(lambda x: (x[0],(x[1][0],x[1][1][1]))) # map current distance (neighbor,([previousNodes1],distance))
			currentSet = currentSet.filter(lambda x: x[1][1] == float('inf')) # check if neighbor has aleady been explored, currentSet should all be (neighbor,([previousNodes1],inf))
			bfsDistances = bfsDistances.partitionBy(10).cache()
			currentSet = currentSet.partitionBy(10).cache()
			count += 1

	return counts

# The code below is the BFS routine when the diameter (10) is used instead of the accumulators

"""

def bfsSearchDiameter(startName, charAdj, sc):

	currentSet = sc.parallelize([startName])
	currentSet = currentSet.map(lambda x: (x,(None,float('inf')))).partitionBy(10).cache()
	characterList = charAdj.map(lambda x: x[0])
	counts = []
	count = 0

	bfsDistances = characterList.map(lambda x: (x,(None,float('inf')))).partitionBy(10).cache()

	def updateDist(node):
		if node[1][1] != None: # value present
			return (node[0],(node[1][1][0],min([node[1][0][1],count])))
		else:
			return (node[0],node[1][0])

	while count < 11: # count is 1 after the root node/start node is initialized to zero
		bfsDistances = bfsDistances.leftOuterJoin(currentSet).cache() # (character,((storedPrevious,storedDist),(setDistPrevious,currentSetDist))), if currentSetDist = none, do nothing 
		bfsDistances = bfsDistances.map(updateDist).cache() # update distances (character,(prevNodes,updatedDistance))
		newValue = bfsDistances.filter(lambda x: x[1][1] != float('inf')).count()
		counts.append(newValue)
		currentSet = currentSet.rightOuterJoin(charAdj)
		currentSet = currentSet.filter(lambda x: x[1][0] != None) # for nodes in current set, get (node, ((None,inf),[node.neighbors]))
		currentSet = currentSet.map(lambda x: (x[0],x[1][1])) # (node,[node.neighbors])
		currentSet = currentSet.map(lambda x: [(ii,x[0]) for ii in x[1]]) #[(neighbor1,node),(neighbor2,node),...]
		currentSet = currentSet.flatMap(lambda x: x).groupByKey().mapValues(list).map(lambda x: (x[0],np.unique(x[1]))) # (neighbor1,[previousNodes1]),(neighbor2,[previousNodes2]),...
		currentSet = currentSet.join(bfsDistances).map(lambda x: (x[0],(x[1][0],x[1][1][1]))) # map current distance (neighbor,([previousNodes1],distance))
		currentSet = currentSet.filter(lambda x: x[1][1] == float('inf')) # check if neighbor has aleady been explored, currentSet should all be (neighbor,([previousNodes1],inf))
		bfsDistances = bfsDistances.partitionBy(10).cache()
		currentSet = currentSet.partitionBy(10).cache()
		count += 1

	return counts

"""

