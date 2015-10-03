import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext(appName="Spark1")

import numpy as np 
import itertools
from P5_bfs_new2 import *

# make spark shut the hell up
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

linklist = sc.textFile('../P5/generated_links_1000_sorted.txt', 32)
titlelist = sc.textFile('../P5/generated_titles_1000_sorted.txt', 32)
#linklist = sc.textFile('../P5/generated_links_10000_sorted.txt', 32)
#titlelist = sc.textFile('../P5/generated_titles_10000_sorted.txt', 32)
#linklist = sc.textFile('../P5/generated_links_100000_sorted.txt', 32)
#titlelist = sc.textFile('../P5/generated_titles_100000_sorted.txt', 32)
#linklist = sc.textFile('../P5/generated_links_2mil_sorted.txt', 32)
#titlelist = sc.textFile('../P5/generated_titles_2mil_sorted.txt', 32)

# Uncomment these when running on AWS@
#linklist = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
#titlelist = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)

numerical_titles = titlelist.zipWithIndex().cache()

num_nodes = numerical_titles.count()
#num_Partitions = int(num_nodes/50)
num_Partitions = 256

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

dist, unreachables = bfs(split_list, start_node, sc, num_Partitions, distances=None, stopNode=end_node)
#print distances.values().countByValue(), '\n'
print dist.take(100)
print "Distance to end node:", dist.map(lambda x: x).lookup(end_node)[0]

print '\n\n\n\n\n\n\n\n\n\n\nCalculating connected components....\n\n'
num_conn = count_connected_components(split_list, num_Partitions, sc)
print '\n\n\n\n\n\nNumber connected components: ', num_conn