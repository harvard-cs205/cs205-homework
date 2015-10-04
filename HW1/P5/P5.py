from pyspark import SparkContext
import numpy as np
from itertools import chain
from P5_bfs import *

sc = SparkContext()
sc.setLogLevel("ERROR")

links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt',32)
pages = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt',32)

#Creating the graph
def mkInt(lst):
	return [int(val) for val in lst]

link_edges = links.map(lambda s: s.split(": ")).map(lambda s: (int(s[0]),s[1]))
link_edges = link_edges.mapValues(lambda l: l.split(" ")).mapValues(list).mapValues(lambda x: mkInt(x)).partitionBy(256).cache()
assoc_pages = pages.zipWithIndex().mapValues(lambda v: v+1).partitionBy(32)

Harvard_ID = assoc_pages.lookup("Harvard_University")[0]
Bacon_ID = assoc_pages.lookup("Kevin_Bacon")[0]


#Run SS-BFS for Harvard > Bacon and Bacon > Harvard
Harv2Bac_dist, Harv2Bac_numTouchedNodes, Harv2Bac_path = P5_bfs(link_edges, Harvard_ID, sc, Bacon_ID)
Bac2Harv_dist, Bac2Harv_numTouchedNodes, Bac2Harv_path = P5_bfs(link_edges, Bacon_ID, sc, Harvard_ID)

print "Harvard to Kevin Bacon: ", Harv2Bac_dist, " connections away through the following path" + '\n' 
print Harv2Bac_path
print '\n\n'
print "Kevin Bacon to Harvard: ", Bac2Harv_dist, " connections away through the following path" + '\n'
print Bac2Harv_path
print '\n\n'