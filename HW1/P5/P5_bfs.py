# References:
# http://hadooptutorial.wikispaces.com/Iterative+MapReduce+and+Counters
# http://www.slideshare.net/jhammerb/lec5-pagerank
# The CSV and Spark manual pages

from pyspark import SparkContext

sc = SparkContext("local", "P5", pyFiles=[])

import csv
import time

INF = 999999999
# issues = {}
# charLookup = {}
# revLookup = []
# adjList = {}
# heroCount = 0
numPart = 24
updates = sc.accumulator(0)

# # Read CSV file
# with open('source.csv','rb') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         name, issue = row[0].strip(), row[1].strip()
#         if issue not in issues:
#             issues[issue] = set()
#         if name not in charLookup:
#             charLookup[name] = heroCount
#             revLookup.append(name)
#             heroCount += 1
#         issues[issue].add(name)
#
# # Create adjacency list
# for cs in issues.values():
#     chars = [charLookup[c] for c in cs]
#     for i in xrange(len(chars)):
#         c1 = chars[i]
#         if c1 not in adjList:
#             adjList[c1] = set()
#         for j in xrange(i+1, len(chars)):
#             c2 = chars[j]
#             if c2 not in adjList:
#                 adjList[c2] = set()
#             adjList[c1].add(c2)
#             adjList[c2].add(c1)

# Map: emit neighbour nodes if GRAY, otherwise
# just emit the same thing if WHITE or BLACK
def processNode((k,v)):
    curDist, adj, col, prevP = v
    ret = []
    if col == 1: # Emit all neighbouring nodes and cur
        for nb in adj:
            ret.append( (nb, (curDist+1, [], 1, prevP+[nb])) )
        ret.append((k, (curDist, adj, 2, prevP)))
    else: # Emit the same thing
        ret.append((k,v))
    return ret

# Combine nodes by key
def combineNode(a, b):
    retAdj = a[1] if len(a[1]) > len(b[1]) else b[1]
    retDist = retPrevP = None
    if a[0] < b[0]:
        retDist = a[0]
        retPrevP = a[3]
    else:
        retDist = b[0]
        retPrevP = b[3]
    retC = max(a[2], b[2]) # Pick the higher color
    return (retDist, retAdj, retC, retPrevP)

# Format of the RDD:
# node_index | current_dist | adj_list | color | path
# color = {WHITE=0, GRAY=1, BLACK=2}
def getPath(graphRDD, startN = 0, destN = 1):
    retP = []
    graphRDD = graphRDD.map(lambda (k,v): (k,v) if (k!=startN) else (k, (0, v[1], 1, [k])), preservesPartitioning=True)
    while True: # While we haven't processed destN and there are still unprocessed nodes
        updates.value = 0
        graphRDD = graphRDD.flatMap(processNode)\
                        .reduceByKey(combineNode)
        # Find the destination node and see if it's been reached
        sel = graphRDD.filter( lambda (k,v): (k==destN and v[0]!=INF) )
        if sel.count() == 1:
            retP = sel.collect()[0][1][3]
            # 'Found destN =', destN,'returning'
            return retP
        # Check we are still iterating
        graphRDD.foreach(lambda (k,v): updates.add(int(v[2]==1)))
        if updates.value == 0:
            # print 'Updating finished, could not find', destN
            break

    return retP

# Input is graphRDD
# Returns (numComponents, maxCompSize)
# Just reusing the functions above, but would be more efficient to not store the path
def findComponents(graphRDD):
    maxCompSize = -1
    numComponents = 0
    while True:
        numComponents += 1
        firstE = graphRDD.take(1)
        # Reference: http://stackoverflow.com/questions/28454357/spark-efficient-way-to-test-if-an-rdd-is-empty
        if not firstE:
            break
        startN = firstE[0][0]
        graphRDD = graphRDD.map(lambda (k,v): (k,v) if (k!=startN) else (k, (0, v[1], 1, [k])), preservesPartitioning=True)
        while True: # While gray nodes still exist
            updates.value = 0
            graphRDD = graphRDD.flatMap(processNode)\
                            .reduceByKey(combineNode)
            graphRDD.foreach(lambda (k,v): updates.add(int(v[2]==1)))
            if updates.value == 0:
                break
        # Count the seen nodes
        curCSize = graphRDD.filter(lambda (k,v): (v[0]!=INF))\
                            .count()
        maxCompSize = max(curCSize, maxCompSize)
        graphRDD = graphRDD.filter(lambda (k,v): (v[0]==INF)) # Select unseen nodes
    return (numComponents, maxCompSize)

# node_index | current_dist | adj_list | color | path
def setupNode(s):
    s = s.split()
    return ( int(s[0][:-1]), (INF, [int(x) for x in s[1:]], 0, []) )

def emitN(s):
    ret = []
    s = s.split()
    cur = int(s[0][:-1])
    nbs = [int(x) for x in s[1:]]
    for nb in nbs:
        ret.append((nb, [cur]))
    ret.append( (cur, nbs) )
    return ret

def onlyBothDir(v):
    ret = []
    count = {}
    for c in v:
        if c not in count:
            count[c] = 1
        else:
            ret.append(c)
    return (k, ret)

def oneDir(v):
    ret = []
    seen = {}
    for c in v:
        if c not in seen:
            ret.append(c)
            seen[c] = True
    return (k, ret)

def getIdx(RDD, s):
    ret = RDD.filter(lambda (k,v): (k==s)).take(1)
    if not ret:
        return -1
    else:
        return ret[0][1] + 1

def resolvePath(RDD, p):
    return [RDD.filter(lambda (k,v): v == i-1).take(1)[0][0] for i in p]

links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
links = links.map(setupNode)

page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt')
page_names = page_names.zipWithIndex().partitionBy(numPart).cache()

# Get the indices for the search
kb_idx = getIdx( page_names, "Kevin_Bacon" )
hu_idx = getIdx( page_names, "Harvard_University" )

# Get the paths for our search
kb2hu = getPath(links, kb_idx, hu_idx)
ret1 = resolvePath(page_names, kb2hu)
print 'Kevin Bacon => Harvard University'
print ret1

hu2kb = getPath(links, hu_idx, kb_idx)
ret2 = resolvePath( page_names, hu2kb )
print 'Harvard University => Kevin Bacon'
print ret2


