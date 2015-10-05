#import findspark
#findspark.init("/home/toby/spark")

import pyspark
from pyspark import AccumulatorParam
def mergeTwoDict(d1, d2):
    res = d2.copy()
    res.update(d1)
    return res

def makeDict(parent, children):
    res = {}
    for child in children:
        res[child] = parent
    return res

def findPath(start, end, graph, sc):

    class PathAccumulatorParam(AccumulatorParam):
        def zero(self, initialValue):
            return [set(), {}]

        def addInPlace(self, v1, v2):
            v1[0] |= v2[0]
            v1[1] = mergeTwoDict(v1[1], v2[1])
            return v1

    visited = sc.accumulator([set(), {}], PathAccumulatorParam())

    print visited.value
    not_visited = set([start])
    iteration = 0

    while(len(not_visited)>0):
        iteration += 1
	    print iteration
        pre = set(list(visited.value[0]))
        graph.filter(lambda (K, V): K in not_visited).foreach(lambda (K, V): visited.add([V, makeDict(K, V)]))
        not_visited = visited.value[0]-pre
        if end in visited.value[0]: break
        
    parents = visited.value[1]

    path = [end]
    while(end!=start):
        path.append(parents[end])
        end = parents[end]
    return path[::-1]


if __name__ == '__main__':
    sc = pyspark.SparkContext(appName="Spark 2")
    sc.setLogLevel('WARN') 

    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
    titles = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)
    titles = titles.zipWithIndex().map(lambda (n, id): (id + 1, n))
    titles = titles.sortByKey().cache()
    #titles = sc.textFile("titles-sorted.txt").collect()
    #links = sc.textFile("links-simple-sorted.txt")

    Kevin_Bacon = titles.filter(lambda (K, V): V == 'Kevin_Bacon').collect()
    # This should be [(node_id, 'Kevin_Bacon')]
    assert len(Kevin_Bacon) == 1
    Kevin_Bacon = Kevin_Bacon[0][0]  # extract node id

    # find Harvard University
    Harvard_University = titles.filter(lambda (K, V): V == 'Harvard_University').collect()
    # This should be [(node_id, 'Harvard_University')]
    assert len(Harvard_University) == 1
    Harvard_University = Harvard_University[0][0]  # extract node id

    def parse_link(line):
        tmp = line.split(": ")
        parent = int(tmp[0])
        children = set([int(node) for node in tmp[1].split()])
        return (parent, children)

    links = links.map(parse_link)
    links = links.partitionBy(256).cache()

    path = findPath(Harvard_University, Kevin_Bacon, links, sc)
    names = []
    for node in path:
        name = titles.filter(lambda (K, V): K==node).collect()
        assert len(name)==1
        name = name[0][1]
        names.append(name)

    print path
    print "->".join(names)

    path = findPath(Kevin_Bacon, Harvard_University, links, sc)
    names = []
    for node in path:
        name = titles.filter(lambda (K, V): K==node).collect()
        assert len(name)==1
        name = name[0][1]
        names.append(name)

    print path
    print "->".join(names)
