# Author: George Lok
# P5_bfs_startup.py

from pyspark import SparkContext, SparkConf

from P4_bfs import *

linksFile = 's3://Harvard-CS205/wikipedia/links-simple-sorted.txt'
titlesFile = 's3://Harvard-CS205/wikipedia/titles-sorted.txt'

# Setup spark
conf = SparkConf().setAppName("CS205_P5a")
sc = SparkContext(conf=conf)

# Tiles are 1-indexed
allTitles = sc.textFile(titlesFile).zipWithIndex().map(lambda (a, b) : (a, b+1))

def lineSplitFunction(line) :
    (a,b) = line.split(':')
    return (int(a), [int(x) for x in b.split()])

allEdges = sc.textFile(linksFile).map(lineSplitFunction).partitionBy(32)

harvard = allTitles.lookup("Harvard_University")[0]
bacon = allTitles.lookup("Kevin_Bacon")[0]

# Approximately 3 minutes on EMR with 2 executors
bToH = SDBFS(bacon, harvard, allEdges)

# Approximately 5 minutes on EMR with 2 executors
hToB = SDBFS(harvard, bacon, allEdges)

numToTitles = allTitles.map(lambda (a,b) : (b,a))

# Uses joins since there can be many unique values, so it's more efficient than lookup
bToHSet= set()
for item in bToH :
    for subitem in item :
        if subitem not in bToHSet :
            bToHSet.add(subitem)

bToHRDD = sc.parallelize(list(bToHSet)).map(lambda x : (x, None))
bToHDict = bToHRDD.join(numToTitles).map(lambda (x, (y,z)) : (x, z)).collectAsMap()
bToHPaths = [[bToHDict[subitem] for subitem in item] for item in bToH]

hToBSet= set()
for item in hToB :
    for subitem in item :
        if subitem not in hToBSet :
            hToBSet.add(subitem)

hToBRDD = sc.parallelize(list(hToBSet)).map(lambda x : (x, None))
hToBDict = hToBRDD.join(numToTitles).map(lambda (x, (y,z)) : (x, z)).collectAsMap()
hToBPaths = [[hToBDict[subitem] for subitem in item] for item in hToB]