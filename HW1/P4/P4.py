from pyspark import SparkContext
from P4_bfs import *


sc = SparkContext("local", "P4")
source = sc.textFile("source.csv")
nodes = source.map(node_KV)
comics = source.map(comic_KV)
comics = comics.reduceByKey(lambda x,y: x + y)
nodes_neighbors = nodes.join(comics).map(get_neighbors).reduceByKey(group_neighbors)
sorted_neighbors = nodes_neighbors.sortByKey()
roots = [u'CAPTAIN AMERICA', u'MISS THING/MARY', u'ORWELL']
for r in roots:
    print r + " " + str(ss_bfs_accum(sorted_neighbors, r))


