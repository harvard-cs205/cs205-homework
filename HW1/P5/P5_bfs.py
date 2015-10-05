# import findspark
# findspark.init()
import pyspark


def build_edges_rdd(initial_rdd, N):
    """Builds the link-link graph."""
    start_rdd = initial_rdd.map(lambda entry: tuple(entry.split(': '))).mapValues(lambda v: v.split(' '))
    start_rdd = start_rdd.map(lambda (k, v): (int(k), v))

    # (Node, Node) format:
    edges_rdd = start_rdd.flatMapValues(lambda v: v).mapValues(lambda v: int(v))
    edges_rdd = edges_rdd.distinct(N).filter(lambda (node1, node2): node1 != node2)  # filter duplicates & links to self
    return edges_rdd  # .sortByKey()


def distance_between(root_node, end_node, edges_rdd, lookup_table, N):
    """Using edge tuples to determine distances to a node, without
       leaving Spark."""
    root = lookup_table.lookup(root_node)[0]
    end = lookup_table.lookup(end_node)[0]
    lookup_table.unpersist()
    edges_rdd = edges_rdd.partitionBy(N)
    edges_rdd = edges_rdd.cache()
    result = edges_rdd.context.parallelize([(root, 0)])
    rdd = result
    while result.filter(lambda (k, v): k == end).isEmpty():
        rdd = edges_rdd.join(rdd).partitionBy(N).values().distinct()
        rdd = rdd.subtractByKey(result)  # don't repeat work
        rdd = rdd.mapValues(lambda v: v + 1)
        result = result.union(rdd).partitionBy(N)
        rdd = rdd.cache()
    return result.filter(lambda (k, v): k == end).map(lambda (k, v): (end_node, v))


def path_between(root_node, end_node, edges_rdd, lookup_table, N):
    """Using edge tuples to find paths between 2 nodes, without
       leaving Spark. Uses two starting points to meet in the middle: saves visiting nodes."""
    root = lookup_table.lookup(root_node)[0]
    end = lookup_table.lookup(end_node)[0]
    lookup_table.unpersist()

    edges_rdd = edges_rdd.partitionBy(N).cache()
    mirror_rdd = edges_rdd.map(lambda (k, v): (v, k)).partitionBy(N).cache()

    paths_to = edges_rdd.context.parallelize([(root, 0)], N)
    paths_fro = edges_rdd.context.parallelize([(end, 0)], N)

    visited = edges_rdd.context.parallelize([(root, 0), (end, 0)], N)

    while paths_to.keys().intersection(paths_fro.keys()).isEmpty():
        print '.',
        # one way
        joins_to = edges_rdd.join(paths_to).partitionBy(N).distinct()
        paths_to = joins_to.map(lambda (k, (v, w)): (v, (k, w)))
        paths_to = paths_to.subtractByKey(visited, N)

        # other way
        joins_fro = mirror_rdd.join(paths_fro).partitionBy(N).distinct()
        paths_fro = joins_fro.map(lambda (k, (v, w)): (v, (k, w)))
        paths_fro = paths_fro.subtractByKey(visited, N)

        # Update blacklist
        visited = visited.union(joins_to.values().map(lambda (v, w): (v, 0))).partitionBy(N)
        visited = visited.union(joins_fro.values().map(lambda (v, w): (v, 0))).partitionBy(N)

    edges_rdd.unpersist()
    mirror_rdd.unpersist()
    print '.'

    # construct paths
    path_parts = paths_to.join(paths_fro).distinct()
    result = path_parts.map(lambda (k, v): list(reversed(unpack_tuple(v[0]))) + [k] + unpack_tuple(v[1]))
    return result


def unpack_tuple(tup):
    result = []
    while True:  # unfortunately Python has no do-while loops
        result += [tup[0]]
        if tup[1] == 0:
            break
        tup = tup[1]
    return result
