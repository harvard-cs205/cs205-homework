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
        next_neighbors = set(current.map(lambda (k, v): v).flatMap(lambda x: x).collect()) - visited
        visited = visited.union(neighbors)
        nodes += len(neighbors)
        current = current.map(lambda (k, v): (k, iteration))
        result = result.union(current)
        neighbors = next_neighbors
        iteration += 1
    print "Number of nodes: " + str(nodes.value)
    return result


def distance_to_all_nodes_edge(root_node, edges_rdd, N):
    """Using edge tuples to determine distances to all nodes, without leaving the Spark"""
    result = edges_rdd.context.parallelize([(root_node, 0)])
    rdd = edges_rdd.filter(lambda (k, v): k == root_node)
    rdd = rdd.map(lambda (k, v): (v, 0))
    while not rdd.isEmpty():
        rdd = rdd.mapValues(lambda v: v + 1)
        result = result.union(rdd)
        rdd = edges_rdd.join(rdd, N).map(lambda (k, (v1, v2)): (v1, v2))
        done = result.join(rdd).map(lambda (k, (v1, v2)): (k, v2))
        rdd = rdd.subtract(done, N)  # don't repeat work
        rdd.repartition(N)
    result.cache()
    result.repartition(N)
    result = result.reduceByKey(lambda n1, n2: min(n1, n2))
    return result


def distance_to_all_nodes_serial(root_node, graph_rdd):
    """Serial implementation that is still pretty fast"""
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
    return result_rdd  # -1 means no connection (within 10 steps)






