import numpy as np
import itertools
from P4_bfs import bfs

import findspark
findspark.init()
import pyspark

sc = pyspark.SparkContext(appName="P4")
sc.setLogLevel('WARN')

comic_data = sc.textFile("source.csv")
comic_data = comic_data.map(lambda x: [a.replace('"','') for a in x.split("\",\"")])
rev = comic_data.map(lambda x: (x[0], x[1]))
newm = rev.map(lambda x: (x[1], [x[0]]))
comic_to_char = newm.reduceByKey(lambda a,b: a + b)

graph = comic_to_char.flatMap(lambda x: itertools.permutations(x[1], 2))

graph = graph.map(lambda x: (x[0], [x[1]]))
adj_list = graph.reduceByKey(lambda a,b: a + b)

print "total number of nodes: ", adj_list.count()

print "Captain America:"
sname = 'CAPTAIN AMERICA'
ret = bfs(adj_list, sname, sc)
s = ret.values().countByValue()
print s, sum(s.values())

print "Miss thing/Mary:"
sname = 'MISS THING/MARY'
ret = bfs(adj_list, sname, sc)
s = ret.values().countByValue()
print s, sum(s.values())

print "Orwell: "
sname = 'ORWELL'
ret = bfs(adj_list, sname, sc)
s = ret.values().countByValue()
print s, sum(s.values())