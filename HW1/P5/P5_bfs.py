# Comment out these lines when running on AWS
import findspark
findspark.init()

import pyspark
sc = pyspark.SparkContext(appName="Spark1")

import numpy as np 
import itertools

def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner

def reducingFun(elem):
    (prev, ((neighbors, _), path)) = elem
    generated_paths = []
    for neighbor in neighbors:
        generated_paths.append((neighbor, path + [neighbor]))
    return generated_paths

#######
# Implements Breadth First Search
# Arguments:
# 	adj (KV RDD): adjacency list - these are directed edges. 
# 	start (string): Where the breadth first search will start
#   sc: the sparkContext
#   numPartitions: number of partitions in RDDs being joined
#   stopNode: where to stop
# Returns:
# 	RDD of paths to stopNode from start
def bfs(adj, start, sc, numPartitions, stopNode):

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
        paths = joined.flatMap(reducingFun).partitionBy(numPartitions).cache()
        # This determines if stopNode has been found.
        paths.filter(lambda (node, _): node == stopNode).foreach(lambda _: solutions.add(1))
        # Get the outlying nodes
        to_search = paths.keys().distinct().map(lambda x: (x, [-1])).partitionBy(numPartitions).cache()


    # Get all of the paths that end with the node we are interested in.
    res = paths.filter(lambda (x, y): x == stopNode).values()
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
linklist = sc.textFile('../P5/generated_links_2mil_sorted.txt', 32)
titlelist = sc.textFile('../P5/generated_titles_2mil_sorted.txt', 32)

# Uncomment these when running on AWS!
#linklist = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
#titlelist = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)

num_Partitions = 256
numerical_titles = titlelist.zipWithIndex().cache()
indexed_titles = numerical_titles.map(lambda (x, y): (y, x)).partitionBy(num_Partitions).cache()
num_nodes = numerical_titles.count()

# Borrowed from Professor's Github example on SparkPageRank
def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) - 1 for to in dests.split(' ')]
    return (int(src) - 1, dests)

def map_node_to_name(arr, reached_to_name):
    ret = []
    for elem in arr:
        ret.append(reached_to_name[elem])
    return ret

split_list = linklist.map(link_string_to_KV).cache()
nodes = split_list.map(lambda (x,y): int(x), True)
assert(copartitioned(split_list, nodes))

# Change these to be the appropriate start and end locations.
start = "Kevin_Bacon"
end = "Harvard_University"

start_node = numerical_titles.lookup(start)[0]
end_node = numerical_titles.lookup(end)[0]

paths = bfs(split_list, start_node, sc, num_Partitions, end_node)
paths = paths.cache()

# Now that we have the paths, we need to get the actual names of the pages associated with those paths. 
# Get the unique items that are reached, join that with the indexed_titles
names_touched = paths.flatMap(lambda x: x).distinct().map(lambda node: (node, node)).partitionBy(num_Partitions).join(indexed_titles).mapValues(lambda (x, y): y)

# Extract a dictionary from the RDD, with key being the node #, and value being the name.
reached_to_name = names_touched.collectAsMap()
# Convert the paths to named paths
named_paths = paths.map(lambda x: map_node_to_name(x, reached_to_name))
print named_paths.collect()




    
