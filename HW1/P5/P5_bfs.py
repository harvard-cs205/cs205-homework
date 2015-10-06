# Initialize Spark
from pyspark import SparkContext
sc = SparkContext()
sc.setLogLevel("ERROR")

# Helper functions for cleaning wiki data
def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) for to in dests.split(' ')]
    return (int(src), dests)

# Helper function for updating path
def get_path(x):
    newChars = x[1][1] 
    oldPath = x[1][0][1]
    return [(c, [len(oldPath + [c]), oldPath + [c]]) for c in newChars]

# Finds the shortest path between two nodes
def find_path(searchGraph, startNodeID, endNodeID):
    
    # Initialize key variables
    searchNodes = sc.parallelize([startNodeID]).map(lambda x: (x, x))
    explored = sc.parallelize([startNodeID]).map(lambda x: (x, x))
    frontier = sc.parallelize([startNodeID]).map(lambda x: (x, [1, [x]]))
    targetNodes = sc.parallelize([])
    dist = 1
    
    # Keep looping until the target node has been found
    while targetNodes.isEmpty():

        dist += 1
        
        # Look up new children nodes
        children = searchGraph.join(searchNodes).map(lambda x: (x[0], x[1][0])).partitionBy(256)
        
        # Update the paths
        frontier = frontier.join(children).map(lambda x: get_path(x)).flatMap(lambda x: x).partitionBy(256)
        
        # Look up all paths that end in the target node
        targetNodes = frontier.filter(lambda x: x[0]==endNodeID).cache()
        
        # Update the list of nodes to search in the next iteration
        searchNodes = frontier.filter(lambda x: x[1][0]==dist).map(lambda x: (x[0], x[0])).distinct().subtractByKey(explored).partitionBy(256).cache()

        # Update the list of explored nodes
        explored = frontier.flatMap(lambda x: x[1][1]).map(lambda x: (x, x)).distinct().partitionBy(256).cache()
        
    return targetNodes.map(lambda x: x[1][1]).collect()

def directed_paths(pageNameGraph, neighborGraph, startNode, endNode):
    
    # find ID of start node
    startNodeDetails = pageNameGraph.filter(lambda (K, V): V == startNode).collect()
    startNodeID = startNodeDetails[0][0]
    
    # find ID of end node
    endNodeDetails = pageNameGraph.filter(lambda (K, V): V == endNode).collect()
    endNodeID = endNodeDetails[0][0]
    
    # Look up shortest paths from start node to end node
    paths = find_path(neighborGraph, startNodeID, endNodeID)

    # Display shortest paths
    print 'All shortest paths:'

    for p in paths:
        path = []
        for n in p:
            path.append(page_names.filter(lambda (K, V): K == n).collect()[0][1])
        print path

############################################################################################
# MAIN FUNCTION
############################################################################################

# Load wiki data & format graph
# Code from https://github.com/thouis/SparkPageRank/blob/master/PageRank.py
links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)
neighbor_graph = links.map(link_string_to_KV)
page_names = page_names.zipWithIndex().map(lambda (n, id): (id + 1, n)).sortByKey().cache()
neighbor_graph = neighbor_graph.partitionBy(256).cache()

# Path search #1
print 'Path search #1: Kevin Bacon >> Harvard University'
directed_paths(page_names, neighbor_graph, 'Kevin_Bacon', 'Harvard_University')

# Path search #2
print '\nPath search #2: Harvard University >> Kevin Bacon'
directed_paths(page_names, neighbor_graph, 'Harvard_University', 'Kevin_Bacon')
