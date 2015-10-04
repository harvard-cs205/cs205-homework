from pyspark import SparkContext

import networkx as nx
import matplotlib.pyplot as plt
import itertools
import pdb

from P4_bfs import bfs

sc = SparkContext()
textFile = sc.textFile('source.csv',20)

#startNode = 'CAPTAIN AMERICA'		
#startNode = 'MISS THING/MARY'
startNode = 'ORWELL'
graph = (
	textFile
	.map(lambda x:(x.split('"')[3],x.split('"')[1]))
	.groupByKey()
	.map(lambda t:(t[0],list(t[1])))
	.flatMap(lambda f:[(y,tuple([x for x in f[1] if x != y])) for y in f[1]])
	.groupByKey().map(lambda f:(f[0],10000,frozenset([inner for outer in f[1] for inner in outer]),False))
	.map(lambda x:(x[0],0,x[2],True) if x[0] == startNode else x)
	.cache()
)

accum = sc.accumulator(0)
accum2 = sc.accumulator(0)
graph = bfs(graph,startNode)
