import findspark
findspark.init('/home/lhoang/spark')

import pyspark
sc = pyspark.SparkContext(appName="spark1")

partition_size = 100
default_distance = 99
sum_distance = sc.accumulator(0)


def l_add_distance(list, d):
    for kv in list:
        yield (kv, d)


def aggregate_distance(d):
    global sum_distance
    if d < default_distance:
        sum_distance += 1


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
        nodes.foreach(lambda kv: aggregate_distance(kv[1]))

        if sum_distance.value - begin_sum_distance == previous_sum_distance:
            print 'Converged after ' + repr(i) + ' iterations'
            break

        root = neighbors.cache()
        i += 1
        previous_sum_distance = sum_distance.value - begin_sum_distance

    return nodes, edges
