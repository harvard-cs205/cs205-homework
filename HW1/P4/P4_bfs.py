import findspark
findspark.init()
import pyspark


# parallel implementation
def distance_to_all_nodes_spark(root_node, graph_rdd):
    graph = graph_rdd.map(lambda (k, v): (k, -1, v))  # set all distances to -1
    neighbors = {root_node}  # initial node
    for iteration in range(11):
        graph = graph.map(lambda (k, n, v): (k, n, v) if (n > -1 or k not in neighbors) else (k, iteration, v))
        visited = graph.filter(lambda (k, n, v): n == iteration)
        neigbors = set(visited.map(lambda (k, n, v): v).reduce(lambda a, b: a + b if b else a))
        if not len(neighbors) == 0:
            visited = visited.map(lambda (k, n, v): (k, n, []))
            graph = graph.subtract(visited)
            graph = graph.union(visited)
        else:
            break  # No neighbors left, done with search
    graph = graph.map(lambda (k, n, v): (k, n))
    return graph


def new(root_node, graph_rdd):
    graph = graph_rdd.map(lambda (k, v): (k, -1, v))  # set all distances to -1
    graph.cache()
    neighbors = graph.context.parallelize(graph_rdd.lookup(root_node))  # initial node
    iteration = graph_rdd.context.accumulator(0)
    while not neighbors.isEmpty():
        checklist = neighbors.collect()[0]
        graph = graph.map(lambda (k, n, v): (k, n, v) if (n > -1 or k not in checklist) else (k, iteration, v))
        visited = graph.filter(lambda (k, n, v): n == iteration)
        neighbors = visited.map(lambda (k, n, v): v).reduce(lambda l1, l2: l1 + l2)
        graph = graph.filter(lambda (k, n, v): n == iteration).map(lambda (k, n, v): (k, n, []))
        iteration += 1
    graph = graph.map(lambda (k, n, v): (k, n))
    return graph


# serial implementation
def distance_to_all_nodes_serial(root_node, graph_rdd):
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






