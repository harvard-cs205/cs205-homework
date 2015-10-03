import findspark
#print findspark.find()
findspark.init()

import pyspark
sc = pyspark.SparkContext(appName="Spark1")
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

from P5_bfs_new import *


#linklist = sc.textFile('../P5/generated_links_representative_sorted.txt', 32)
#titlelist = sc.textFile('../P5/generated_titles_representative_sorted.txt', 32)
linklist = sc.textFile('../P5/generated_links_small_sorted.txt', 32)
titlelist = sc.textFile('../P5/generated_titles_small_sorted.txt', 32)
numerical_titles = titlelist.zipWithIndex().cache()

num_nodes = numerical_titles.count()
#num_Partitions = int(num_nodes/50)
num_Partitions = 128

# Borrowed from Professor's Github example on SparkPageRank
def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) - 1 for to in dests.split(' ')]
    return (int(src) - 1, dests)

split_list = linklist.map(link_string_to_KV).cache()
nodes = split_list.map(lambda (x,y): int(x), True)
assert(copartitioned(split_list, nodes))

#start = "Kevin_Bacon"
#end = "Harvard_University"
start = 'TITLE_8'
end = "TITLE_4"

start_node = numerical_titles.lookup(start)[0]
end_node = numerical_titles.lookup(end)[0]
print start_node, end_node

distances, unreachables = bfs(split_list, start_node, sc, num_Partitions, nodes, end_node)
#print distances.values().countByValue(), '\n'
print distances.take(100)
print "Distance to end node:", distances.map(lambda x: x).lookup(end_node)[0]