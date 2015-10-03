from operator import add
from operator import itemgetter
from itertools import groupby
import time

import findspark
findspark.init('/home/lhoang/spark')

import pyspark
sc = pyspark.SparkContext(appName="spark1")

partition_size = 20
default_distance = 99999
sum_distance = sc.accumulator(0)


def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner


def l_get_symmetric_pairs(kv):
    neighbors = kv[1].split()
    for nb in neighbors:
        yield (kv[0], nb)
        yield (nb, kv[0])


def l_get_pairs(kv):
    neighbors = kv[1].split()
    for nb in neighbors:
        yield (kv[0], nb)


def quiet_logs(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)
    logger.LogManager.getLogger("amazonaws").setLevel(logger.Level.WARN)


def create_symmetric_graph(fileName):
    """
    Create a symmetric graph of wiki pages.
    :param fileName: The file to load.
    """
    src = sc.textFile(fileName)

    # make symmetric links then reduce to give a list of
    # (page_i, [neighbors of page_i])
    edges = src.map(lambda x: x.split(':')).flatMap(
        l_get_symmetric_pairs).distinct().reduceByKey(
        add).map(lambda kv: (kv[0], list(set(kv[1])))).partitionBy(
        partition_size).cache()

    # create list of nodes with initial distance = 99
    nodes = edges.mapValues(lambda v: default_distance).cache()

    return nodes, edges


def create_dual_graph(fileName):
    """
    Create a dual graph of wiki pages.
    :param fileName: The file to load.
    """
    src = sc.textFile(fileName)

    # make dual links then reduce to give a list of
    # (page_i, [neighbors of page_i])
    edge_pairs = src.map(lambda x: x.split(':')).flatMap(
        l_get_pairs)
    edge_pairs_reverse = edge_pairs.map(lambda kv: (kv[1], kv[0]))

    edges = edge_pairs.intersection(edge_pairs_reverse).reduceByKey(
        add).map(lambda kv: (kv[0], list(set(kv[1])))).partitionBy(
        partition_size).cache()

    # create list of nodes with initial distance = 99
    nodes = edges.mapValues(lambda v: default_distance).cache()

    return nodes, edges


def l_add_distance(list, d):
    for kv in list:
        yield (kv, d)


def update_accumulator(d):
    global sum_distance
    if d < default_distance:
        sum_distance += 1


def bfs_search(nodes, edges, page):
    """
    Perform breadth-first search to find shortest path from root.
    :param page: The root page to find paths from.
    """
    root = nodes.filter(lambda kv: kv[0] == page)

    i = 1
    previous_sum_distance = 0
    while (True):
        begin_sum_distance = sum_distance.value

        # get neighbors and make sure they are copartitioned with edges & nodes
        # since each partition may contain duplicate keys, we use mapPartitions
        # to eliminate these without causing a shuffle.
        neighbors = edges.join(root).flatMap(
            lambda kv: l_add_distance(kv[1][0], i)).distinct().partitionBy(
              partition_size)

        nodes = neighbors.union(nodes).reduceByKey(min)

        # check for convergence using accumulator
        nodes.foreach(lambda kv: update_accumulator(kv[1]))

        if sum_distance.value - begin_sum_distance == previous_sum_distance:
            print 'Converged after ' + repr(i) + ' iterations'
            break

        root = neighbors
        i += 1
        previous_sum_distance = sum_distance.value - begin_sum_distance

    return nodes, edges


def find_components():
    # nodes, edges = create_symmetric_graph('links.smp')
    nodes, edges = create_dual_graph('links.smp')
    root = nodes.first()[0]

    global sum_distance

    num_components = 0
    num_total_elements = nodes.count()
    num_total_component_elements = 0
    while True:
        sum_distance += -sum_distance.value
        nodes_bfs, _ = bfs_search(nodes, edges, root)

        num_components += 1
        print '------------- COMPONENT ' + repr(num_components)
        print '------------- ROOT = ' + repr(root)

        num_elements = nodes_bfs.filter(
            lambda kv: kv[1] < default_distance).count()
        print '# of connected elements = ' + repr(num_elements)

        num_total_component_elements += num_elements
        if num_total_component_elements >= num_total_elements:
            break

        untouched_nodes = nodes_bfs.filter(
            lambda kv: kv[1] == default_distance)

        if untouched_nodes.isEmpty():
            break

        # get the first node that hasn't been touched to search again
        root = untouched_nodes.first()[0]
    pass
