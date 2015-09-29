from operator import add
from operator import itemgetter
from itertools import groupby
import time

import findspark
findspark.init('/home/lhoang/spark')

import pyspark
sc = pyspark.SparkContext(appName="spark1")

partition_size = 100
default_distance = 99


def create_graph(fileName):
    """
    Create a graph of character relationship given a file.
    :param fileName: The csv file to load where each line = (hero, comic).
    """
    src = sc.textFile(fileName)

    # convert records to (comic, hero)
    vk = src.map(lambda x: x.split('"')).map(lambda tup: (tup[3], tup[1]))

    # join records by comic to find which heroes are related
    g = vk.join(vk).map(lambda kv: sorted(list(set(list(kv[1]))))).filter(
        lambda seq: len(seq) > 1)

    # create unique links (hero_i, hero_j)
    links = g.map(lambda seq: (seq[0], seq[1])).distinct()

    # make links symmetric by mapping (hero_i, hero_j) to (hero_j, hero_i)
    # then reduce to give a list of (hero_i, [neighbors of hero_i])
    edges = links.flatMap(lambda kv: [kv, (kv[1], kv[0])]).map(
        lambda kvp: (kvp[0], [kvp[1]])).reduceByKey(
        add).map(lambda kv: (kv[0], list(set(kv[1])))).partitionBy(
        partition_size).cache()

    # create list of nodes with initial distance = 99
    nodes = edges.mapValues(lambda v: default_distance)

    return nodes, edges


def p_reduce_min(part):
    for k, g in groupby(part, key=itemgetter(0)):
        lis = [y for x in g for y in x[1:]]
        yield (k, min(lis))


def p_get_keys(part):
    for kv in part:
        yield kv[0]


def p_get_partition_keys_with_index(index, part):
    for kv in part:
        yield (index, kv[0])


def p_get_unique_keys(part):
    keys = []
    for kv in part:
        keys.append(kv[0])
    keys = list(set(keys))
    return keys


def p_get_unique_keys_with_distance(part, d):
    kd_pairs = []
    for kv in part:
        kd_pairs.append((kv[0], d))
    kd_pairs = list(set(kd_pairs))
    return kd_pairs


def p_distinct(part):
    return list(set(part))


def l_add_distance(list, d):
    for kv in list:
        yield (kv, d)


def bfs_search(nodes, edges, hero, diameter):
    """
    Perform breadth-first search to find shortest path from hero.
    :param hero: The root node to find paths from.
    """
    # initially root nodes only contains the source node
    # but will grow to contain more nodes at each iteration
    root = nodes.filter(lambda kv: kv[0] == hero)

    # iteratively find distances for other nodes
    for i in range(1, diameter + 1):
        # get neighbors and make sure they are copartitioned with edges & nodes
        # since each partition may contain duplicate keys, we use mapPartitions
        # to eliminate these without causing a shuffle.

        # neighbors = edges.join(
        #   root).flatMap(lambda kv: l_add_distance(kv[1][0], i)).partitionBy(
        #     partition_size).mapPartitions(
        #     lambda part: p_distinct(part))
        neighbors = edges.join(root).flatMap(
            lambda kv: l_add_distance(kv[1][0], i)).distinct().partitionBy(
              partition_size)

        nodes = neighbors.union(nodes).reduceByKey(min)
        # nodes = neighbors.union(nodes).mapPartitions(
        #     lambda part: p_reduce_min(part))

        root = neighbors

    return nodes, edges


# def bfs_search_iteration(i, nodes, edges, root):
#     """
#     Perform breadth-first search at iteration i.
#     :param i: The iteration.
#     :param nodes: list of nodes (hero, distance).
#     :param edges: list of edges (hero_i, hero_j).
#     :param root: list of nodes (hero_i) whose neighbors will be updated.
#     """
#     print 'i = ' + repr(i)

#     neighbors = root.map(lambda n: (n, 0)).join(edges).flatMap(
#         lambda kv: kv[1][1]).distinct()

#     updated_nodes = neighbors.map(lambda n: (n, i)).join(nodes).map(
#         lambda kv: (kv[0], kv[1][0]) if kv[1][0] != -1 else (
#             kv[0], kv[1][1]))

#     nodes = updated_nodes.union(nodes).reduceByKey(lambda d1, d2: max(d1, d2))

#     return nodes, edges, neighbors


def main():
    nodes, edges = create_graph('source.csv')

    diameter = 10
    num_display_neighbors = 10

    root = ['CAPTAIN AMERICA', 'MISS THING/MARY', 'ORWELL']

    for hero in root:
        nodes_bfs, _ = bfs_search(nodes, edges, hero, diameter)

        start = time.time()
        print '---------------------'
        print hero

        for i in range(3):
            print 'Showing ' + repr(num_display_neighbors) + ' D-' + repr(
                i + 1) + ' neighbors:'
            print nodes_bfs.filter(
                lambda kv: kv[1] == i + 1).take(num_display_neighbors)

        print 'Number of touched nodes:'
        print nodes_bfs.filter(lambda kv: kv[1] != default_distance).count()

        print 'Running time: '
        print (time.time() - start)

    sc.stop()
