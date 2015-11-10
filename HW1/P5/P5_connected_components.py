# References:
# http://hadooptutorial.wikispaces.com/Iterative+MapReduce+and+Counters
# http://www.slideshare.net/jhammerb/lec5-pagerank
# The CSV and Spark manual pages

from pyspark import SparkContext

sc = SparkContext(appName="P5")

INF = 999999999

numPart = 24
updates = sc.accumulator(0)

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
    retDist = retPrevP = None
    if a[0] < b[0]:
        retDist = a[0]
    else:
        retDist = b[0]
    retC = max(a[2], b[2]) # Pick the higher color
    return (retDist, retAdj, retC)


# Input is graphRDD
# Returns (numComponents, maxCompSize)
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
        graphRDD = graphRDD.map(lambda (k,v): (k,v) if (k!=startN) else (k, (0, v[1], 1)), preservesPartitioning=True)
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

# node_index | current_dist | adj_list | color
def setupNode_CC(v):
    return (INF, v, 0)

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
    return (INF, ret, 0)

def oneDir(v):
    ret = []
    seen = {}
    for c in v:
        if c not in seen:
            ret.append(c)
            seen[c] = True
    return (INF, ret, 0)


links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt')
allN = links.flatMap(emitN).reduceByKey(lambda a, b: a+b).partitionBy(numPart).cache()
links_undir = allN.mapValues(onlyBothDir)
links_dir = allN.mapValues(oneDir)

# print links_undir.take(1)
# print links_dir.take(1)


f = open('out2.txt','w')
res = findComponents(links_undir)
print 'Connected Components:', res
f.write(str(res))
f.close()

f = open('out3.txt','w')
res = findComponents(links_dir)
print 'Connected Components:', res
f.write(str(res))
f.close()


# rddList = []
#
# for node, adj in adjList.iteritems():
#     dist = INF
#     rddList.append( (node, (dist, list(adj), 0, []) ) )
#
# graphRDD = sc.parallelize(rddList, numPart)\
#                 .cache()

# sT = time.time()
# sList = [('CAPTAIN AMERICA', 'MISS THING/MARY'), ('LOBO', 'WOODGOD'), ('WOODGOD', 'TYRUS'), ('URICH, BEN', 'VIPER II'),
#         ('HAWK', 'JARVIS, EDWIN')]
# for chars in sList:
#     sN, dN = chars
#     if sN not in charLookup or dN not in charLookup:
#         print 'One of',chars,'not found!'
#         continue
#     retP = getPath(graphRDD, charLookup[sN], charLookup[dN])
#     print 'Source =', sN,'Destination=', dN
#     if not retP:
#         print 'No path found!'
#     else:
#         print 'Path =', [revLookup[idx] for idx in retP]
# print time.time() - sT, 'seconds'

# Find connected components for both symmetric graph and strictly symmetric graph

# Do BFS with touched nodes
# Repeat while there are still nodes
    # Remove all touched nodes
    # Count size
    # Increment counter of numComponents

# sT = time.time()
# print 'Connected Components:', findComponents(graphRDD)
# print time.time() - sT, 'seconds'
