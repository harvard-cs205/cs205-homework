import time
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from P4_BFS import *
from  pyspark import SparkContext
sc = SparkContext()
sc.setLogLevel("ERROR")

# Load the source file
text = sc.textFile("source.txt")

#Break each line to form tuples: (Comic issues, Superhero) 
rdd1 = text.map(lambda r: (r.split('",')[1].replace('"',''),r.split('",')[0].replace('"','')))

#Create every possible pair of superheroes appearing in the same comic issue (and removing the tuple mapping each superhero to himself)
edgelist = rdd1.join(rdd1).map(lambda (K,V): V).filter(lambda (K,V): K!=V).distinct().sortByKey().partitionBy(8)

#Forcing the actual computation of the edglist and caching it:
edgelist.take(1)
edgelist.cache()

print "Edgelist is ready, starting computing"

start2 = time.time()
missmary = bfs(edgelist,'MISS THING/MARY',10,sc)
end2 = time.time()

print '\n'
print "MISS THING/MARY:\n"
print end2-start2
print missmary[0]
print '\n'

start3 = time.time()
orwell = bfs(edgelist,'ORWELL',10,sc)
end3 = time.time()

print "ORWELL:\n"
print end3-start3
print orwell[0]
print '\n'

start1 = time.time()
captain = bfs(edgelist,'CAPTAIN AMERICA',10,sc)
end1 = time.time()

print "CAPTAIN AMERICA:\n"
print end1-start1
print captain[0]
print '\n'