import findspark
findspark.init()
import pyspark


def distance_to_all_nodes_spark(root_node, graph_rdd):
    """A fast parallel implementation of the search algorithm (does leave RDD for neighbors, for performace reasons)"""
    result = graph_rdd.context.parallelize([(root_node, 0)])
    visited = {root_node}  # set initial node
    neighbors = set(graph_rdd.lookup(root_node)[0])   # set initial neighbors
    nodes = graph_rdd.context.accumulator(1)
    iteration = 1
    while neighbors:  # we are done when neigbors is empty
        current = graph_rdd.filter(lambda (k, v): k in neighbors)
        current = current.mapValues(lambda v: iteration)
        new_neighbors = set(current.flatMap(lambda x: x).collect())
        result = result.union(current)
        nodes += len(neighbors)
        visited = visited.union(neighbors)
        neighbors = new_neighbors - visited
        iteration += 1
    print "Number of nodes: " + str(nodes.value)
    return result


def distance_to_all_nodes_edge(root_node, edges_rdd, N):
    """Using edge tuples to determine distances to all nodes, without leaving the Spark"""
    edges_rdd.cache()
    result = edges_rdd.context.parallelize([(root_node, 0)])
    rdd = edges_rdd.filter(lambda (k, v): k == root_node)
    rdd = rdd.map(lambda (k, v): (v, 0))
    while not rdd.isEmpty():
        rdd = edges_rdd.join(rdd, N).values()
        rdd = rdd.subtractByKey(result, N)  # don't repeat work
        rdd = rdd.mapValues(lambda v: v + 1)
        result = result.union(rdd)
        rdd = rdd.partitionBy(N).cache()
    return result

if __name__ == '__main__':
    pass