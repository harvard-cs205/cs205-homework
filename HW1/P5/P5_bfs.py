from pyspark import AccumulatorParam

def copartitioned(RDD1, RDD2):
    return RDD1.partitioner == RDD2.partitioner

def _bfs(graph, source):
    sc = graph.context
    result = graph.filter(lambda (k,v): k == source).mapValues(lambda _: 0)
    assert copartitioned(result, graph)
    frontier = result.mapValues(lambda _: 0)
    assert copartitioned(result, frontier)
    numPtt = result.getNumPartitions()

    resultSize, newResultSize = 1, 0

    def bfsMapper(x):
        d, adjlist = x[1]
        return [(adj, d+1) for adj in adjlist]

    while resultSize != newResultSize:
        resultSize = newResultSize
        frontier = frontier.join(graph) \
                            .flatMap(bfsMapper)

        result = result.union(frontier).reduceByKey(lambda d1, d2: min(d1, d2), numPartitions=numPtt)
        assert copartitioned(result, graph)
        # enforce lazy-evaluation
        newResultSize = result.count()
        result.cache()
        frontier.cache()

    return result 

class SetAccm(AccumulatorParam):
    def zero(self, s):
        return s

    def addInPlace(self, s1, s2):
        s1 |= s2
        return s1

def shortestPath(graph, source, end):
    sc = graph.context
    childParent = sc.parallelize([(source, -1)])

    frontier, closed = set([source]), set([source])

    while True:
        tmpFoundAccm = sc.accumulator(set(), SetAccm())
        toSearch = graph.filter(lambda (k, _): k in frontier)
        toSearch.foreach(lambda (k, adj): tmpFoundAccm.add(set(adj)))

        curChildParent = toSearch.flatMap(lambda (p, chdr): [(c, p) for c in chdr])
        curChildParent = curChildParent.reduceByKey(lambda p1, p2: p1)
        curChildParent = curChildParent.filter(lambda (c, p): c not in closed)
        childParent = childParent.union(curChildParent)

        frontier = tmpFoundAccm.value - closed
        if len(frontier) == 0:
            return None

        closed |= tmpFoundAccm.value
        
        if end in frontier:
            dist, chdToFind = 0, end
            path = [chdToFind]

            while True:
                entry = childParent.filter(lambda (k, _): k == chdToFind).collect()
                if len(entry) == 0:
                    return None
                chdToFind = entry[0][1]
                if chdToFind == -1:
                    break

                path.append(chdToFind)
                dist += 1
            path = path[::-1]
            return path, dist
    return None


def connected(graph):
    keyList = graph.map(lambda kv: (kv[0], 0))
    count = 0
    while True:
        #keyList.cache()
        #keyListCount = keyList.count()
        #print "Key list count: ", keyListCount
        #if keyListCount == 0:
        #    break
        try:
            source = keyList.take(1)[0][0]
            #print source
            # very important! O.W. keyList.getNumPartitions() will grow gradually
            # keyList = keyList.repartition(max(NUM_PTT_LOWER, min(keyList.getNumPartitions()+1, NUM_PTT_UPPER)))
            components = _bfs(graph, source)
            count += 1
            keyList = keyList.subtractByKey(components, components.getNumPartitions())
            keyList.cache()

            print 'Count for this iter is {0}'.format(count)

        except Exception, e:
            print 'Exception occured, count is {0}'.format(count)
            print str(e)
            break
    return count
