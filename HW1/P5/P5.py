import time

import pyspark
sc = pyspark.SparkContext(appName="spark1")

partition_size = 20
default_distance = 99999
sum_distance = sc.accumulator(0)


def quiet_logs(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)
    logger.LogManager.getLogger("amazonaws").setLevel(logger.Level.WARN)


def create_graph(fileName, nameFile):
    """
    Create a graph of character relationship given a file.
    :param fileName: The csv file to load where each line = (hero, comic).
    """
    src = sc.textFile(fileName)

    # make links symmetric by mapping (hero_i, hero_j) to (hero_j, hero_i)
    # then reduce to give a list of (hero_i, [neighbors of hero_i])
    edges = src.map(lambda x: x.split(':')).map(
        lambda kv: (int(kv[0]), [int(v) for v in kv[1].split()])).partitionBy(
        partition_size).cache()

    # create list of nodes with initial distance = 99
    nodes = edges.mapValues(lambda v: (default_distance, ''))

    names = sc.textFile(nameFile).zipWithIndex().map(
        lambda kv: (kv[1] + 1, kv[0])).partitionBy(
            partition_size).cache()

    return nodes, edges, names


def l_add_distance(kv, d):
    for v in kv[1][0]:
        yield (v, (d, kv[1][1][1] + ' ' + repr(kv[0])))


def update_accumulator(d):
    global sum_distance
    if d[0] < default_distance:
        sum_distance += 1


def l_min_distance(v1, v2):
    if v1[0] < v2[0]:
        return v1
    else:
        return v2


def bfs_search(nodes, edges, names, source, target):
    """
    Perform breadth-first search to find shortest path from source.
    :param source: The root node to find paths from.
    """
    # initially root nodes only contains the source node
    # but will grow to contain more nodes at each iteration
    root = names.filter(lambda kv: kv[1] == source).join(
        nodes).mapValues(lambda kv: kv[1])

    end = names.filter(lambda kv: kv[1] == target).cache()

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

        # stop if target has been found
        if end.join(nodes).filter(
                lambda kv: kv[1][0] < default_distance).isEmpty() == False:
            print 'Found target'
            break

        # check for convergence using accumulator
        nodes.foreach(lambda kv: update_accumulator(kv[1]))

        if sum_distance.value - begin_sum_distance == previous_sum_distance:
            print 'Converged after ' + repr(i) + ' iterations'
            break

        root = neighbors
        i += 1
        previous_sum_distance = sum_distance.value - begin_sum_distance

    root_end_path = end.join(nodes)

    return nodes, edges, root_end_path


def bfs(links_file, titles_file, root, target):
    start = time.time()

    nodes, edges, names = create_graph(links_file, titles_file)
    _, _, root_end_path = bfs_search(nodes, edges, names, root, target)

    path = root_end_path.flatMap(lambda kv: kv[1][1][1].split()).map(
        lambda v: (int(v), 0)).join(names).map(lambda kv: kv[1][1]).collect()

    print '-------- FOUND PATH --------'
    print path

    print 'TIME ' + repr(time.time() - start)

quiet_logs(sc)

bfs('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt',
    's3://Harvard-CS205/wikipedia/titles-sorted.txt',
    'Kevin_Bacon',
    'Harvard_University')

bfs('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt',
    's3://Harvard-CS205/wikipedia/titles-sorted.txt',
    'Harvard_University',
    'Kevin_Bacon')
