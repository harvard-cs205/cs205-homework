import pyspark
#from P5_bfs import *
from pyspark import AccumulatorParam
import time

def _buildDirectedGraph(links):
    # links: RDD
    dirGraph = links.map(lambda line: line.split()) \
                    .map(lambda lst: (lst[0][:-1], lst[1:]))
    return dirGraph

# Build the symmetric graph
def _buildSymmGraph(links):
    # links: RDD
    dirGraph = _buildDirectedGraph(links)
    symmGraph = dirGraph.flatMap(lambda pair: [(to, [pair[0]]) for to in pair[1]])
    
    symmGraph = symmGraph.union(dirGraph) \
                        .reduceByKey(lambda l1, l2: l1+l2) \
                        .map(lambda kv: (kv[0], list(set(kv[1]))))
    return symmGraph

# Build the required graph which satisfies the requirement that
# edge exists only if a pair is connected in both direction.
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

# Accumulator used for BFS. Basically the algorithm adds all the visited
# nodes in one iteration into this accumulator instance.
class SetAccm(AccumulatorParam):
    def zero(self, s):
        return s

    def addInPlace(self, s1, s2):
        s1 |= s2
        return s1

def bfs(graph, source):
    sc = graph.context
    childParent = sc.parallelize([(source, -1)])

    frontier, closed = set([source]), set([source])

    while True:
        # Initialize the accumulator in each iteration
        tmpFoundAccm = sc.accumulator(set(), SetAccm())
        # Those entries in frontier queue will be searched
        toSearch = graph.filter(lambda (k, _): k in frontier)
        # Add all the adjacent nodes into the accumulator
        toSearch.foreach(lambda (k, adj): tmpFoundAccm.add(set(adj)))

        frontier = tmpFoundAccm.value - closed
        if len(frontier) == 0:
            return None

        closed |= tmpFoundAccm.value
    return closed

def connected(graph):
    keyList = graph.map(lambda kv: (kv[0]))
    count = 0
    while True:
        try:
            source = keyList.take(1)[0]
            closed = bfs(graph, source)
            count += 1
            # Remove those that are visited
            keyList = keyList.filter(lambda k: k not in closed)
            keyList.cache()

            print 'Count for this iter is {0}'.format(count)

        except Exception, e:
            print 'Exception occured, count is {0}'.format(count)
            print str(e)
            break
    return count

if __name__ == '__main__':
    sc = pyspark.SparkContext(appName='YK-P5')
    
    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
    #graph = _buildSymmGraph(links)
    graph = _buildGraph2(links)
    graph.cache()
    
    connCount = connected(graph2)
    
    print 'Connected: {0}'.format(connCount)
    with open('graph2_connected.txt', 'w') as fh:
        fh.write(str(connCount))