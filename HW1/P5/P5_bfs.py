# import findspark
# findspark.init()
import pyspark


def build_edges_rdd(initial_rdd, N):
    """Builds the link-link graph """
    start_rdd = initial_rdd.map(lambda entry: tuple(entry.split(': '))).mapValues(lambda v: v.split(' '))
    start_rdd = start_rdd.map(lambda (k, v): (int(k), v))

    # (Node, Node) format:
    edges_rdd = start_rdd.flatMapValues(lambda v: v).mapValues(lambda v: int(v))
    edges_rdd = edges_rdd.distinct(N).filter(lambda (node1, node2): node1 != node2)  # filter duplicates & links to self
    return edges_rdd  # .sortByKey()


def distance_between(root_node, end_node, edges_rdd, lookup_table, N):
    """Using edge tuples to determine distances to a node, without leaving Spark"""
    root = lookup_table.lookup(root_node)[0]
    end = lookup_table.lookup(end_node)[0]
    edges_rdd = edges_rdd.partitionBy(N)
    edges_rdd = edges_rdd.cache()
    result = edges_rdd.context.parallelize([(root, 0)])
    rdd = result
    while result.filter(lambda (k, v): k == end).isEmpty():
        rdd = edges_rdd.join(rdd).partitionBy(N).values()
        rdd = rdd.distinct()
        rdd = rdd.subtractByKey(result)  # don't repeat work
        rdd = rdd.mapValues(lambda v: v + 1)
        result = result.union(rdd).partitionBy(N)
        rdd = rdd.cache()
    return result.filter(lambda (k, v): k == end).map(lambda (k, v): (end_node, v))