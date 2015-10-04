import time
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from P4_BFS import *
from  pyspark import SparkContext
sc = SparkContext()
sc.setLogLevel("ERROR")

text = sc.textFile() 
rdd1 = text.map(lambda r: (r.split('",')[1].replace('"',''),r.split('",')[0].replace('"','')))
rdd2 = rdd1.join(rdd1).map(lambda (K,V): V).filter(lambda (K,V): K!=V).distinct()
network =  rdd2.groupByKey().map(lambda (K,V): (K,list(V)))

edgelist = network.flatMap(lambda (K,V): [(K,V[i]) for i in range(len(V))]).sortByKey()
edgelist.take(1) 
edgelist.partitionBy(8)
edgelist.cache()

start2 = time.time()
missmary = bfs2(edgelist,'MISS THING/MARY',10,sc)
end2 = time.time()

print '\n'
print "MISS THING/MARY:\n"
print end2-start2
print missmary[0]
print '\n'

start3 = time.time()
orwell = bfs2(edgelist,'ORWELL',10,sc)
end3 = time.time()

print "ORWELL:\n"
print end3-start3
print orwell[0]
print '\n'

start1 = time.time()
captain = bfs2(edgelist,'CAPTAIN AMERICA',10,sc)
end1 = time.time()

print "CAPTAIN AMERICA:\n"
print end1-start1
print captain[0]
print '\n'