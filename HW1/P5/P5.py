import pyspark
#from P5_bfs import *
from pyspark import AccumulatorParam
import time

def _buildDirectedGraph(links):
    # links: RDD
    dirGraph = links.map(lambda line: line.split()) \
                    .map(lambda lst: (lst[0][:-1], lst[1:]))
    return dirGraph

def _buildSymmGraph(links):
    # links: RDD
    dirGraph = _buildDirectedGraph(links)
    symmGraph = dirGraph.flatMap(lambda pair: [(to, [pair[0]]) for to in pair[1]])
    
    symmGraph = symmGraph.union(dirGraph) \
                        .reduceByKey(lambda l1, l2: l1+l2) \
                        .map(lambda kv: (kv[0], list(set(kv[1]))))
    return symmGraph

def _buildGraph2(links):
    dirGraph = _buildDirectedGraph(links)
    keyList = dirGraph.flatMap(lambda kv: [(k, []) for k in [kv[0]] + kv[1]])
    
    dirGraph = dirGraph.flatMap(lambda pair: [(pair[0], to) for to in pair[1]])
    revGraph = dirGraph.map(lambda pair: (pair[1], pair[0]))
    
    graph2 = dirGraph.intersection(revGraph) \
                    .map(lambda pair: (pair[0], [pair[1]])) \
                    .union(keyList) \
                    .map(lambda kv: (kv[0], list(set(kv[1])))) \
                    .reduceByKey(lambda v1, v2: v1 + v2)
    return graph2

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
    
def findIndices(names, nameIndexMap):
    entries = nameIndexMap.filter(lambda (n, i): n in set(names)).collect()
    if len(entries):
        return {e[0]:e[1] for e in entries}
    return None

def findNames(indices, indexNameMap):
    entries = indexNameMap.filter(lambda (i, n): i in set(indices)).collect()
    if len(entries):
        return {e[0]:e[1] for e in entries}
    return None

if __name__ == '__main__':
    sc = pyspark.SparkContext(appName='YK-P5')
    
    #links = sc.textFile('links-simple-sorted.txt')
    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
    pageNames = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')
    nameIndexMap = pageNames.zipWithIndex().map(lambda (k, v): (k, str(v+1))).cache()
    indexNameMap = nameIndexMap.map(lambda (k, v): (v, k)).cache()

    #links = sc.textFile('test.csv')
    dirGraph = _buildDirectedGraph(links)

    #dirGraph = _buildDirectedGraph(sc.textFile('test.csv'))
    dirGraph = dirGraph.repartition(256)
    dirGraph.cache()

    end, source = 'Harvard_University', 'Kevin_Bacon'
    indices = findIndices([source, end], nameIndexMap)
    
    sourceIdx = indices[source]
    endIdx = indices[end]

    result = shortestPath(dirGraph, sourceIdx, endIdx)
    if result is not None:
        path, dist = result
        names = findNames(path, indexNameMap)
        namedPath = []
        for i in path:
            namedPath.append(names[i])

        with open('{0}-{1}'.format(source, end), 'w') as fh:
            fh.write('Source: {0}, Index: {1}\n'.format(source, sourceIdx))
            fh.write('End: {0}, Index: {1}\n'.format(end, endIdx))
            fh.write(' '.join(namedPath) + '\n')
            fh.write('Dist: {0}\n'.format(dist))
    #print result

    #symmGraph = _buildSymmGraph(links)
    #graph = sc.parallelize(graph)
    
    #print "Connected components: {0}".format(connected(graph))
    #end = '2152782'
    #source = '2729536'
    #minDist = shortestPath(dirGraph, source, end)

    #graph2 = _buildGraph2(links)
    #graph2.cache()
    
    #connCount = connected(graph2)
    '''
    lines = graph2.collect()
    for ln in lines:
        print ln
    '''
    #print 'Connected: {0}'.format(connCount)
    #with open('graph2_connected.txt', 'w') as fh:
    #    fh.write(str(connCount))
    # time.sleep(5)
    #print '{0}-{1}: {2}'.format(source, end, result)
    #_writeDistDict(distDict, 'CAA') 