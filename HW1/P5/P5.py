# P5 solution

sourcefile = 'links-simple-sorted.txt'
lookupfile = 'titles-sorted.txt'

from P5_bfs import *

from pyspark import SparkConf, SparkContext
conf = SparkConf().setAppName('KaiSquare')
sc = SparkContext(conf = conf)

if __name__  == '__main__':

    lookuplist = sc.textFile(lookupfile)  # build the look up title list
    lookuplist = lookuplist.zipWithIndex().map(lambda kv: (kv[0], kv[1]+1))  # zip with index, from title to number
    lookdownlist = lookuplist.map(lambda kv: (kv[1], kv[0]))  # from number to title

    emptylist = lookuplist.map(lambda kv: (kv[1], []))  # empty list is needed since possible isolated node

    wikilist = sc.textFile(sourcefile)
    wikilist = wikilist.map(lambda line: line.split())  # line comprehension
    wikilist = wikilist.map(lambda line: (line[0][:-1], line[1:]))

    wikilist = emptylist.subtractByKey(wikilist).union(wikilist)  # if not empty, create the real adjacent list
    wikilist = wikilist.partitionBy(64)
    wikilist.cache()

    forward = lambda line: [(line[0], v) for v in line[1]]  # functions to buildundirected graphs
    backward = lambda line: [(v, line[0]) for v in line[1]]

    forwikilist = wikilist.flatMap(forward)
    backwikilist = wikilist.flatMap(backward)
    
    biwikilist = forwikilist.intersection(backwikilist)  # if both directions
    omniwikilist = forwikilist.union(backwikilist)  # if either direction

    biwikilist = biwikilist.reduceByKey(lambda a,b: a+b)
    omniwikilist = omniwikilist.reduceByKey(lambda a,b: a+b)
    biwikilist = biwikilist.map(lambda c: (c[0], list(set(c[1]))))  # both direction graph
    omniwikilist = omniwikilist.map(lambda c: (c[0], list(set(c[1]))))  # either direction graph
    biwikilist = biwikilist.partitionBy(64)
    omniwikilist = omniwikilist.partitionBy(64)
    biwikilist.cache()
    omniwikilist.cache()

    namestart = "Kevin_Bacon"  # target nodes
    nameend = "Harvard_University"

    nodestart = str(lookuplist.lookup(namestart)[0])  # lookup to index
    nodeend = str(lookuplist.lookup(nameend)[0])

    # the shortest path part
    print nodestart, nodeend  # the indices
    dist = shortestpath(nodestart, nodeend, wikilist)
    print namestart, nameend
    print dist[0], [lookdownlist.lookup(int(node)) for node in dist[1]]
    distrev = shortestpath(nodeend, nodestart, wikilist)
    print nameend, namestart
    print distrev[0], [lookdownlist.lookup(int(node)) for node in distrev[1]]

    # the connected component part
    print 'bothway:', connectedcomp(biwikilist)
    print 'eitherway:', connectedcomp(omniwikilist)
