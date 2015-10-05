from pyspark import SparkContext
from Queue import Queue

MAX_DIAM = 10

# takes a root node, graph as adjacency list RDD
def rdd_bfs(root, graph):
    # keep track of which nodes are on the current depth
    neigh_set = [root]
    # keep track of distances of nodes visited
    dist = {}
    dist[root] = 0

    # use diam to limit depth of search
    diam = 0
    while neigh_set and diam < MAX_DIAM:
        neighbor_graph = graph.filter(lambda (k, v): k in neigh_set)
        neigh_list = neighbor_graph.flatMap(lambda (k, v): v).collect()
        neigh_set = list(set(neigh_list).difference(set(dist.keys())))
        diam += 1
        for n in neigh_set:
            dist[n] = diam

    return dist
