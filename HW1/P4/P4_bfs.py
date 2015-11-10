# References:
# http://hadooptutorial.wikispaces.com/Iterative+MapReduce+and+Counters
# http://www.slideshare.net/jhammerb/lec5-pagerank
# The CSV and Spark manual pages

INF = 999999999

# Map: emit neighbour nodes if GRAY, otherwise
# just emit the same thing if WHITE or BLACK
def processNode((k,v)):
    curDist, adj, col = v
    ret = []
    if col == 1: # Emit all neighbouring nodes and cur
        for nb in adj:
            ret.append( (nb, (curDist+1, [], 1)) )
        ret.append((k, (curDist, adj, 2)))
    else: # Emit the same thing
        ret.append((k,v))
    return ret

# Combine nodes by key
def combineNode(a, b):
    retAdj = a[1] if len(a[1]) > len(b[1]) else b[1]
    retDist = min(a[0], b[0])
    retC = max(a[2], b[2]) # Pick the higher color

    return (retDist, retAdj, retC)

# Format of the RDD:
# node_index | current_dist | adj_list | color
# color = {WHITE=0, GRAY=1, BLACK=2}
def nTouched(sc, adjList, startN = 0):
    updates = sc.accumulator(0)
    rddList = []

    for node, adj in adjList.iteritems():
        dist = 0 if node == startN else INF
        rddList.append( (node, (dist, list(adj), int(dist==0)) ) )

    graphRDD = sc.parallelize(rddList, 2)
    while True: # While gray nodes still exist
        updates.value = 0
        graphRDD = graphRDD.flatMap(processNode)\
                        .reduceByKey(combineNode)

        graphRDD.foreach(lambda (k,v): updates.add(int(v[2]==1)))
        if updates.value == 0:
            break

    touched = graphRDD.map(lambda (k,v): int(v[0]!=INF), preservesPartitioning=True)\
                    .reduce(lambda a,b: a+b)
    return touched