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
    currentNodes = nodeMap.filter(lambda node: node[0] == rootNode).cache() # cache
    # initialize set of "touched" nodes
    connectedNodes = currentNodes.map(lambda node: (node[0], 0))    
    # initialize the distance
    dist = 0
    # cannot pass an RDD through rdd.filter, so create a list of touched nodes
    l = connectedNodes.map(lambda node: node[0]).collect()
    # keep searching until the size of the map stops changing  
    while (mapSize.value > prevSize):
        # update prevSize 
        prevSize = mapSize.value
        # update distance
        dist += 1
        # update currentNodes to the next "depth" of nodes
        currentNodes = currentNodes.flatMap(lambda node: node[1]).distinct()
        # update currentNodes to contain only the new nodes (minimize bandwidth)
        currentNodes = currentNodes.filter(lambda node: node not in l)
        # update the accumulator (and cache currentNodes because we use it twice more)
        currentNodes.cache().foreach(lambda node: mapSize.add(1))
        # update the node list
        l += currentNodes.collect()
        # update the connectivity map RDD
        connectedNodes = currentNodes.map(lambda node: (node, dist)).union(connectedNodes)
        # prepare currentNodes for the next iteration
        currentNodes = nodeMap.join(currentNodes.map(lambda node: (node, {}))).map(lambda node: (node[0], node[1][0]))
    # return the size of the world (but could return other constituents as well)
    return mapSize.value
