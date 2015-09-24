import findspark
findspark.init()
import pyspark
import numpy as np


def distance_to_all_nodes_from(root_node, graph_rdd):
    result = {}
    current_nodes = [root_node]
    for iteration in xrange(11):  # max 10 iterations
        neighbors = []
        for node in current_nodes:
            if node not in result:
                result[node] = iteration
                neighbors += graph_rdd.lookup(node)[0]
        current_nodes = neighbors
    print len(result)  # number of nodes touched on
    result_rdd = graph_rdd.keys().map(lambda key: (key, -1) if key not in result else (key, result[key]), True)
    return result_rdd.sortBy(lambda (k, v): v, False)  # -1 means no connection (within 10 steps)

