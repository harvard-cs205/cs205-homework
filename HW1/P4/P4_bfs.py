# """ P4_bfs.py """
from pyspark import SparkContext
from pyspark import AccumulatorParam

# note: map is an RDD consisting of (element, set([elements])), representing connectivity in a graph
# with accumulators, no restriction on world size (keep searching until the map converges), minimizing bandwidth
def bfs2(nodeMap, rootNode):
    # initialize the size of the world (1 node)
    mapSize = nodeMap.context.accumulator(1)
    # initialize the "previous count"
    prevSize = 0
    # initialize set of current nodes (rdd.lookup would give a tuple, not an RDD) (note: nodeMap "should" be cached from P4.py)
    currentNodes = nodeMap.filter(lambda node: node[0] == rootNode)
    # initialize set of "touched" nodes
    connectedNodes = currentNodes.map(lambda node: (node[0], 0)).cache() # cache
    # initialize the distance
    dist = 0
    # keep searching until the size of the map stops changing  
    while (mapSize.value > prevSize):
        # update prevSize 
        prevSize = mapSize.value
        # update distance
        dist += 1
        # update currentNodes to the next "depth" of nodes
        currentNodes = currentNodes.flatMap(lambda node: node[1]).distinct()
        # map currentNodes to contain distances, then perform a LOJ with connectedNodes
        currentNodes = currentNodes.map(lambda node: (node, dist)).leftOuterJoin(connectedNodes) # LOJ shuffle
        # filter currentNodes based on LOJ, then map back to (node, dist) pairs
        currentNodes = currentNodes.filter(lambda (K, V): V[1] == None).map(lambda (K, V): (K, V[0])).cache() # cache
        # update the accumulator
        currentNodes.foreach(lambda node: mapSize.add(1))
        # update the connectivity map RDD
        connectedNodes = currentNodes.union(connectedNodes).cache() # cache
        # prepare currentNodes for the next iteration
        currentNodes = nodeMap.join(currentNodes.map(lambda (K, V): (K, {}))).map(lambda node: (node[0], node[1][0])) # join shuffle
    # return the size of the world (but could return other constituents as well)
    return mapSize.value
