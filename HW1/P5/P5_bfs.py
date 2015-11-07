##############
# Ankit Gupta
# ankitgupta@college.harvard.edu
# CS 205, Problem Set 1, P6
#
# Implementation of BFS
#
# Note that I have kept some lines (commented out) to show how I tested this locally. Please ignore those if running on AWS.
# I have included comments that explain what files I used to test this locally before running the real one on AWS. 
##############

# Comment out these lines when running on AWS
#import findspark
#findspark.init()

import pyspark
sc = pyspark.SparkContext(appName="Spark1")

import numpy as np 
import itertools

# Checks if the two RDDs are copartioned.
#   Ex: assert(copartitioned(rdd1, rdd2))
def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner

# This function is used to generate paths.
def pathGeneratingReducer(elem):
    (prev, ((neighbors, _), path)) = elem
    generated_paths = []
    for neighbor in neighbors:
        generated_paths.append((neighbor, path + [neighbor]))
    return generated_paths

# Borrowed from Professor's Github example on SparkPageRank
def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) - 1 for to in dests.split(' ')]
    return (int(src) - 1, dests)

##
# Maps a node number to the associated title.
# Arguments:
#   arr: A list of nodes numbers
#   reached_to_name: A dict that maps node numbers of nodes that the BFS reached to their name
# Returns:
#   List containing the named nodes
def map_node_to_name(arr, reached_to_name):
    ret = []
    for elem in arr:
        ret.append(reached_to_name[elem])
    return ret

##
# Given the graph and start and end nodes, returns the paths (named with titles) from start node to end node.
# Arguments:
#   adj: The graph in adjacency list representation
#   start_node: The node # where BFS is starting
#   end_node: The node # where BFS is ending
def get_paths(adj, start_node, sc, num_Partitions, end_node, indexed_titles):
    paths = bfs(adj, start_node, end_node, sc, num_Partitions)
    paths = paths.cache()
    names_touched = paths.flatMap(lambda x: x).distinct().map(lambda node: (node, node)).partitionBy(num_Partitions).join(indexed_titles).mapValues(lambda (x, y): y)
    reached_to_name = names_touched.collectAsMap()
    named_paths = paths.map(lambda x: map_node_to_name(x, reached_to_name))
    return named_paths.collect()

##
# Implements Breadth First Search
# Arguments:
# 	adj (KV RDD): adjacency list - these are directed edges. 
# 	start (string): Where the breadth first search will start
#   end: where to stop
#   sc: the sparkContext
#   numPartitions: number of partitions in RDDs being joined
#   
# Returns:
# 	RDD of paths to end from start
def bfs(adj, start, end, sc, numPartitions):

    solutions = sc.accumulator(0)

    # adj is the graph, to_search is the outlying nodes, and path contains nodes along with the path to get to those nodes.
    adj = adj.map(lambda x: x).partitionBy(numPartitions).cache()
    to_search = adj.filter(lambda (x, y): x == start).partitionBy(numPartitions)
    paths = to_search.map(lambda (x, neighbors): (x, [x])).partitionBy(numPartitions)


    # Repeat this until the first iteration that we find a solution.
    #   Note that this process may lead to non-shortest paths for nodes that are closer than the one we are looking for, but that's okay since we only care about the one we are going to!
    while solutions.value == 0:
        assert(copartitioned(adj, to_search))
        assert(copartitioned(adj, paths))
        searchers = adj.join(to_search)
        joined = searchers.join(paths)

        # THis looks something like (node, [path]), (node, [path]) ....
        paths = joined.flatMap(pathGeneratingReducer).partitionBy(numPartitions).cache()
        # This determines if end has been found.
        paths.filter(lambda (node, _): node == end).foreach(lambda _: solutions.add(1))
        # Get the outlying nodes
        to_search = paths.keys().distinct().map(lambda x: (x, [-1])).partitionBy(numPartitions).cache()


    # Get all of the paths that end with the node we are interested in.
    res = paths.filter(lambda (x, y): x == end).values()
    return res


# Reduce the amount that Spark logs to console
logger = sc._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )


# Uncomment the appropriate set of files when running locally!
#   These are some of the files I generated in order to run tests locally before running on AWS
#   The number in the file names indicates the number of nodes. Each node has an average of 25 out links (same as true dataset)
#linklist = sc.textFile('../P5/generated_links_1000_sorted.txt', 32)
#titlelist = sc.textFile('../P5/generated_titles_1000_sorted.txt', 32)
#linklist = sc.textFile('../P5/generated_links_10000_sorted.txt', 32)
#titlelist = sc.textFile('../P5/generated_titles_10000_sorted.txt', 32)
#linklist = sc.textFile('../P5/generated_links_2mil_sorted.txt', 32)
#titlelist = sc.textFile('../P5/generated_titles_2mil_sorted.txt', 32)
#linklist = sc.textFile('../P5/links-simple-sorted.txt', 32)
#titlelist = sc.textFile('../P5/titles_sorted.txt', 32)

# Uncomment these when running on AWS!
linklist = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
titlelist = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)

num_Partitions = 256
numerical_titles = titlelist.zipWithIndex().cache()
indexed_titles = numerical_titles.map(lambda (x, y): (y, x)).partitionBy(num_Partitions).cache()
num_nodes = numerical_titles.count()



graph = linklist.map(link_string_to_KV).cache()
nodes = graph.map(lambda (x,y): int(x), True)
assert(copartitioned(graph, nodes))

# Change these to be the appropriate start and end locations.
bacon = "Kevin_Bacon"
harvard = "Harvard_University"

# These were test titles for when running locally.
#bacon = 'TITLE_a'
#harvard = 'TITLE_c'

bacon_node = numerical_titles.lookup(bacon)[0]
harvard_node = numerical_titles.lookup(harvard)[0]

print get_paths(graph, bacon_node, sc, num_Partitions, harvard_node, indexed_titles), '\n\n'
print get_paths(graph, harvard_node, sc, num_Partitions, bacon_node, indexed_titles)




    
