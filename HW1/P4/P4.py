from pyspark import SparkContext
import numpy as np
from itertools import chain
from P4_bfs import *

sc = SparkContext("local[8]", appName="HW1-4 Marvel Graph")
sc.setLogLevel("ERROR")

#Creating the graph
rdd = sc.textFile('source.csv').map(lambda line: line.split(",")).map(lambda line: [line[-1],''.join(line[:-1])])
rdd_grouped = rdd.groupByKey().mapValues(list).map(lambda x: x[1])
HeroConnections_byIssue = rdd_grouped.map(lambda x: [(a,[k for k in x if k is not a]) for a in x]).flatMap(lambda x: x)
HeroConnections_adjacency = HeroConnections_byIssue.groupByKey().mapValues(list).map(lambda x: (x[0],list(set(chain.from_iterable(x[1])))) )
HeroConnections_edges = HeroConnections_adjacency.flatMapValues(lambda x: x).partitionBy(20).cache()

#Run SS-BFS for "CAPTAIN AMERICA", "MISS THING/MARY" and "ORWELL"
ca_numNodesVisited, ca_searchedDist = P4_bfs(HeroConnections_edges, u'"CAPTAIN AMERICA"', sc)
mt_numNodesVisited, mt_searchedDist = P4_bfs(HeroConnections_edges, u'"MISS THING/MARY"', sc)
or_numNodesVisited, or_searchedDist = P4_bfs(HeroConnections_edges, u'"ORWELL"', sc)