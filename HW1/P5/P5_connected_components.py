# import findspark
# findspark.init()
import pyspark


def distance_to_all_nodes_edge(root_node, edges_rdd, N):
    """Using edge tuples to determine distances to all nodes, without leaving Spark."""
    edges_rdd = edges_rdd.partitionBy(N)
    edges_rdd = edges_rdd.cache()
    result = edges_rdd.context.parallelize([(root_node, 0)])
    rdd = result
    while not rdd.isEmpty():
        print '.',
        rdd = edges_rdd.join(rdd).partitionBy(N).values().distinct()
        rdd = rdd.subtractByKey(result)  # don't repeat work
        rdd = rdd.mapValues(lambda v: v + 1)
        result = result.union(rdd)
        rdd = rdd.cache()
    return result


def connected_components(edges_rdd, N):
    """Returns number of connected components with their sizes."""
    component_sizes = []
    while not edges_rdd.isEmpty():
        print '.',
        node = edges_rdd.takeSample(False, 1)[0]
        component = distance_to_all_nodes_edge(node, edges_rdd, N)
        edges_rdd = edges_rdd.subtractByKey(component).partitionBy(N)
        component_sizes += [component.count()]
    print '.'
    return component_sizes
