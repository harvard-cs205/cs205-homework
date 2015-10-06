# """ P5_bfs.py """
from pyspark import SparkContext
from pyspark import AccumulatorParam

# gives a list of nodes based on input (list of nextNodes, source) -> [(nextNode1, source), (nextNode2, source), ...]
def initNode(K, V):
    l = []
    for pg in K:
        l.append((pg, V))
    return l

# note: map is an RDD consisting of (element, [elements]), representing connectivity in a graph
# with accumulators, no restriction on world size (keep searching until the map converges)
def bfs2(nodeMap, source, destination, partitions):
    # initialize the size of the world (1 node)
    mapSize = nodeMap.context.accumulator(1)
    # initialize the "previous count"
    prevSize = 0
    # initialize the distance
    dist = 0
    # initialize currentNodes as an RDD with one node, corresponding to source    
    currentNodes = nodeMap.filter(lambda node: node[0] == source) # (source, [pages conneted to source])
    # initialize connectedNodes as an RDD containing all nodes that have been touched
    connectedNodes = currentNodes.map(lambda node: node[0])
    # using initNode, update currentNodes to contain the next node
    currentNodes = currentNodes.flatMap(lambda (K, V): initNode(V, K))
    # update connectedNodes to include newly touched nodes
    connectedNodes = currentNodes.map(lambda (K, V): K).union(connectedNodes)
    # keep searching until the size of the map stops changing
    while (mapSize.value > prevSize & dist <= 10):
        # update prevSize 
        prevSize = mapSize.value
        # update distance
        dist += 1
        # test if destination is in currentNodes
        testNode = currentNodes.lookup(destination)
        # if test is successful, then exit the loop and return the set of paths and the distance
        if (testNode != []):
            return [(destination, testNode[0])], dist
        # update the accumulator
        currentNodes.foreach(lambda node: mapSize.add(1))
        # for each node in currentNodes, join with nodeMap to obtain the set of nextNodes for each currentNode
        currentNodes = currentNodes.join(nodeMap).map(lambda (K, V): (V[1], (K, V[0])))
        # update currentNodes to contain the full list of nextNodes
        currentNodes = currentNodes.flatMap(lambda (K, V): initNode(K, V))
        # group currentNodes by key to avoid collisions
        currentNodes = currentNodes.groupByKey().mapValues(list)
        # remove the set of nextNodes that have already been seen (to filter out redundancy)
        currentNodes = currentNodes.subtractByKey(connectedNodes.map(lambda node: (node, []))).cache()
        # update the set of touched nodes
        connectedNodes = currentNodes.map(lambda (K, V): K).union(connectedNodes).cache()
    # return a distance of -1 if the entire connected map was searched (but destination was not found)
    return [], -1

# wrapper function for the recursive function, used to obtain node paths
def sortOutput(tup, rdd):
    path, dist = tup
    if dist == -1:
        return "No Connection Found"
    else:
        return rec(path[0], dist, rdd)

# returns the title given an index and the indexedTitles map
def indexToTitle(index, rdd):
    return rdd.map(lambda (K, V): (V, K)).lookup(index)[0]

# recursive function that outputs a list of ordered sequences representing the possible node paths
def rec(paths, dist, rdd):
    if dist == 1:
        return [[indexToTitle(paths[1], rdd), indexToTitle(paths[0], rdd)],]
    lastElem = indexToTitle(paths[0], rdd)
    p =[]
    for path in paths[1]:
        temp = rec(path, dist-1, rdd)
        for elem in temp:
            p.append(elem + [lastElem])
    return p
      

if __name__ == '__main__':
    # initialize the spark context
    sc = SparkContext(appName = "Wikipedia")
    # set log level to 'WARN' when on AWS
    sc.setLogLevel('WARN')
    # set number of partitions to 256
    partitions = 256
    # read in links and titles
    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
    titles = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')
    # reorganize the links to (link, [(list of connected links])
    keyedLinks = links.map(lambda link: tuple(link.split(": "))).map(lambda (K, V): (K, V.split(" "))).cache()
    # index the titles
    indexedTitles = titles.zipWithIndex().map(lambda (K, V): (K, str(V + 1))).sortByKey().cache()
    # find connectivity between two links:
    titlePair = ['Harvard_University', 'Kevin_Bacon']
    # initialize vector that will contain indices representative of above
    indexPair = []
    # obtain indices of elements in titlePair and place them in indexPair
    for link in titlePair:
        indexPair.append(indexedTitles.lookup(link)[0])
    # print all paths from A to B using a single-source breadth-first search
    print "From " + titlePair[0] + " to " + titlePair[1] + ";"
    l1 = sortOutput(bfs2(keyedLinks, indexPair[0], indexPair[1], partitions), indexedTitles)
    for path1 in l1:
        print path1
    # print all paths from B to A using a single-source breadth-first search
    print "\n\nFrom " + titlePair[1] + " to " + titlePair[0] + ":"
    l2 = sortOutput(bfs2(keyedLinks, indexPair[1], indexPair[0], partitions), indexedTitles)
    for path2 in l2:
        print path2
