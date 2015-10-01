# import findspark
# findspark.init()
import pyspark


def build_edges_rdd(initial_rdd):
    start_rdd = initial_rdd.map(lambda entry: tuple(entry.split(': '))).mapValues(lambda v: v.split(' '))
    start_rdd = start_rdd.map(lambda (k, v): (int(k), v))

    # (Node, Node) format:
    edges_rdd = start_rdd.flatMapValues(lambda v: v).mapValues(lambda v: int(v))
    edges_rdd = edges_rdd.distinct(N).filter(lambda (node1, node2): node1 != node2)  # filter duplicates & links to self
    return edges_rdd.sortByKey(numPartitions=N)


def distance_between(root_node, end_node, edges_rdd, lookup_table):
    """Using edge tuples to determine distances to all nodes, without leaving Spark"""
    root = lookup_table.lookup(root_node)[0]
    end = lookup_table.lookup(end_node)[0]
    edges_rdd = edges_rdd.partitionBy(N)
    edges_rdd = edges_rdd.cache()
    result = edges_rdd.context.parallelize([(root, 0)])
    rdd = result
    while not result.lookup(end):
        rdd = edges_rdd.join(rdd).values().partitionBy(N)
        rdd = rdd.distinct()
        rdd = rdd.subtractByKey(result)  # don't repeat work
        rdd = rdd.mapValues(lambda v: v + 1)
        result = result.union(rdd)
        rdd = rdd.cache()
    return result.filter(lambda (k, v): k == end).map(lambda (k, v): (end_node, v))


if __name__ == '__main__':
    N = 64  # Number of partitions
    sc = pyspark.SparkContext("local[32]")
    sc.setLogLevel("ERROR")

    # Get files
    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', N, use_unicode=False)
    page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', N, use_unicode=False)

    # Build RDDs
    edge_rdd = build_edges_rdd(links)
    lookup_table = page_names.zipWithIndex().mapValues(lambda v: v + 1)  # 1-indexed

    # Distance between nodes in network
    distance = distance_between("Kevin_Bacon", "Harvard_University", edge_rdd, lookup_table).collect()
    print distance

