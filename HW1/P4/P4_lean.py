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
sum_distance = sc.accumulator(0)


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


def l_add_distance(list, d):
    for kv in list:
        yield (kv, d)


def update_accumulator(d):
    global sum_distance
    sum_distance += d


def bfs_search(nodes, edges, hero, diameter):
    """
    Perform breadth-first search to find shortest path from hero.
    :param hero: The root node to find paths from.
    """
    # initially root nodes only contains the source node
    # but will grow to contain more nodes at each iteration
    root = nodes.filter(lambda kv: kv[0] == hero)

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
