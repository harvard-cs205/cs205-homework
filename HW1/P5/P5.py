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

    # make links symmetric by mapping (hero_i, hero_j) to (hero_j, hero_i)
    # then reduce to give a list of (hero_i, [neighbors of hero_i])
    edges = src.map(lambda x: x.split(':')).map(
        lambda kv: (kv[0], kv[1].split())).partitionBy(
        partition_size).cache()

    # create list of nodes with initial distance = 99
    nodes = edges.mapValues(lambda v: (default_distance, ''))

    return nodes, edges


def l_add_distance(kv, d):
    for v in kv[1][0]:
        yield (v, (d, kv[1][1][1] + ' ' + kv[0]))


def update_accumulator(d):
    global sum_distance
    sum_distance += d[0]


def l_min_distance(v1, v2):
    if v1[0] < v2[0]:
        return v1
    else:
        return v2


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
            lambda kv: l_add_distance(kv, i)).distinct().partitionBy(
              partition_size)

        nodes = neighbors.union(nodes).reduceByKey(l_min_distance)

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
    nodes, edges = create_graph('links.smp')

    diameter = 10
    num_display_neighbors = 10

    root = ['1', '2', '5']

    for hero in root:
        nodes_bfs, _ = bfs_search(nodes, edges, hero, diameter)

        start = time.time()
        print '---------------------'
        print hero

        for i in range(5):
            print 'Showing ' + repr(num_display_neighbors) + ' D-' + repr(
                i + 1) + ' neighbors:'
            print nodes_bfs.filter(
                lambda kv: kv[1][0] == i + 1).take(num_display_neighbors)

        print 'Number of touched nodes:'
        print nodes_bfs.filter(lambda kv: kv[1][0] != default_distance).count()

        print 'Running time: '
        print (time.time() - start)

    sc.stop()
