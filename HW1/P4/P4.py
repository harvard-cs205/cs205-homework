import findspark
findspark.find()
findspark.init(edit_profile=True)
import pyspark
sc = pyspark.SparkContext()

from functools import partial
import time
from P4_bfs import BFS

# Helper function, used to generate a list of (key, value) tuples given a node set
def generatePairs(node_set):
    edges = []
    for i in node_set:
        for j in node_set:
            if(i != j):
                edges.append((i, j))
    return edges

rawData = sc.textFile("source.csv").map(lambda line: line.split('","')).map(lambda x: [x[0]+'"', '"'+x[1]]).cache()

# Get the list of all characters
charactersRDD = rawData.map(lambda x: x[0]).distinct()

graphRDD = charactersRDD.map(lambda x: (x, []))

# Get a list of connected characters, each element integrate all the characters in a single issue
# [[name1, name2,... ],...,[name_n, name_m]]
edges_tmp = rawData.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: list(x[1]))

# Generate MC_graph based on edges_tmp, here the MC_graph only contains nodes that are connected 
MC_Graph = edges_tmp.flatMap(generatePairs).groupByKey().mapValues(lambda x: list(set(x)))

# filter out connected nodes in charactersRDD and get isolated nodes
connected_nodes = MC_Graph.keys().collect()
isolated_nodes = charactersRDD.filter(lambda K: K not in connected_nodes).map(lambda x: (x, []))

#Finalize MC_Graph by unioning both connected and isolated nodes
MC_Graph = MC_Graph.union(isolated_nodes).cache()

# apply BFS algorithm to get the number of touched nodes in each search

# CAPTAIN AMERICA: 6408
ca_touched_count = BFS(MC_Graph, 'CAPTAIN AMERICA')
# MISS THING/MARY: 7
mtm_touched_count = BFS(MC_Graph, 'MISS THING/MARY')
# ORWELL: 9
o_touched_count = BFS(MC_Graph, 'ORWELL')
