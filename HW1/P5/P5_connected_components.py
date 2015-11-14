# Author: George Lok
# P5_connected_components.py

from pyspark import SparkContext, SparkConf
import time
from operator import add

# N.B. I was unable find enough time to optimize the connectedComponents algorithm so that it would run in a reasonable time.
# One issue I noticed upon running this program is that an iteration takes a significant amount of time (longer than an iteration of BFS)
# Most likely, this is caused by the way I mark and handle the next nodes to explore.

# On the marvel graph, the code works however.

# Setup spark
conf = SparkConf().setAppName("CS205_P5b")
sc = SparkContext(conf=conf)

def connectedComponents(graph) :
    def makeMarkFlag(nCid):
        def markFlag( (source, (cid, flag)) ) :
            if cid == nCid :
                return (source, (cid, 1))
            else :
                return (source, (cid, flag))
        return markFlag
    def reduceFunction(dest1, dest2) :
        return set(dest1) | set(dest2)
    def reduceKeyFunction(cid1, cid2) :
        return min(cid1, cid2)
    def makeMapFunction(dests, newCid) :
        def mapFunction(  (source, (cid, flag)) ) :
            if source in dests :
                if not flag :
                    return (source, (newCid, flag))
            return (source, (cid, flag))    
        return mapFunction
    graph = graph.cache()

    # Create a cidMap, which contains (source, (c[omponent]Id, [visited]flag))
    cidMap = graph.keys().zipWithUniqueId().map(lambda (x,y) : (x,(y,0))).cache()
    counter = 0
    startTime = time.time()
    while(True) :
        print "Beginning Iteration: " + str(counter)
        # Don't look at nodes we've already visited
        filteredCidMap = cidMap.filter(lambda (x, (y,f)) : (f == 0) )

        # All nodes have been visited
        if filteredCidMap.count() == 0 :
            break 
        filteredCidMap = filteredCidMap.map(lambda (x, (y,f)) : (x,y) )

        # Grab the node with the smallest cid that we haven't visted yet
        (_, nextCid) = filteredCidMap.min(key=(lambda (x,y) : y))

        # Grab all nodes with the same cid that we haven't visted yet
        nextSources = filteredCidMap.filter(lambda (x, y) : (y == nextCid) )

        # Mark all nodes we're about to jump from
        cidMap = cidMap.map(makeMarkFlag(nextCid))
        start = time.time()

        # Get next set of nodes we're going to visit
        nextDests = nextSources.join(graph).flatMap(lambda (source, (cid, dests)) : [dests]).reduce(reduceFunction)

        # Mark new nodes with their new cid
        cidMap = cidMap.map(makeMapFunction(nextDests,nextCid)).cache()
        end = time.time()

        print "Iteration Time: " + str(end - start)
        counter += 1
    endTime = time.time()

    print "Total Time : " + str(endTime + startTime)
    print "Iteration Complete: " + str(counter)

    # Now, group by cid and then return just the components.
    return cidMap.map(lambda (source, (cid, flag)) : (cid,source) ).groupByKey().map(lambda (x,y) : list(y))

#  We don't care about titles as per problem spec
linksFile = 's3://Harvard-CS205/wikipedia/links-simple-sorted.txt'


def lineSplitFunction(line) :
    (a,b) = line.split(':')
    return (int(a), [int(x) for x in b.split()])


outEdges = sc.textFile(linksFile).map(lineSplitFunction).partitionBy(32).cache()


def flatMap((source, dests)) :
    results = []
    for dest in dests :
        results.append((dest, source))
    return results


inEdges = outEdges.flatMap(flatMap).groupByKey().partitionBy(32)


bothEdges = outEdges.join(inEdges).partitionBy(32).cache()

def allMap((node, (outList, inList))) :
    return (node, list(set(outList + list(inList))))


def sharedMap((node, (outList, inList))) :
    return (node,  list(set(outList) & set(inList)) )


# Case where all directed edges are now undirected
allEdges = bothEdges.map(allMap)

sharedEdges = bothEdges.map(sharedMap)

# Find connectedComponents for allEdges
allConnectedComponents = connectedComponents(allEdges)

# Find connectedComponents for allEdges
sharedConnectedComponents = connectedComponents(sharedEdges)








