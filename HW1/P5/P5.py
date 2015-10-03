import time

import pyspark
sc = pyspark.SparkContext(appName="spark1")

partition_size = 20
default_distance = 99999
sum_distance = sc.accumulator(0)


def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner


def l_add_distance(ids_distance_path, d):
    for page_id in ids_distance_path[0]:
        yield (page_id, (d, ids_distance_path[1][1] + ' ' + repr(page_id)))


def update_accumulator(d):
    global sum_distance
    if d[0] < default_distance:
        sum_distance += 1


def l_min_distance(v1, v2):
    if v1[0] < v2[0]:
        return v1
    else:
        return v2


def quiet_logs(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)
    logger.LogManager.getLogger("amazonaws").setLevel(logger.Level.WARN)


def create_graph(fileName, nameFile):
    """
    Create a graph of character relationship given a file.
    :param fileName: The csv file to load where each
    line = (page: [page, ...]).
    """
    src = sc.textFile(fileName)

    # make links symmetric by mapping (page_i, page_j) to (page_j, page_i)
    # then reduce to give a list of (page_i, [neighbors of page_i])
    edges = src.map(lambda x: x.split(':')).map(
        lambda kv: (int(kv[0]), [int(v) for v in kv[1].split()])).partitionBy(
        partition_size).cache()

    # create list of nodes with initial distance = 99
    # [ (id, (distance, 'path')) ]
    nodes = edges.map(lambda kv: (kv[0], (
        default_distance, repr(kv[0]))), preservesPartitioning=True).cache()

    # RDD mapping from 1-based index to page names
    # [ (index, name) ]
    names = sc.textFile(nameFile).zipWithIndex().map(
        lambda kv: (kv[1] + 1, kv[0])).partitionBy(
            partition_size).cache()

    return nodes, edges, names


def bfs_search(nodes, edges, names, source, target):
    """
    Perform breadth-first search to find shortest path from source.
    :param source: The root node to find paths from.
    """
    # get the source node to find path from
    # [ (index, (distance, 'path')) ]
    root = names.filter(lambda kv: kv[1] == source).join(
        nodes).mapValues(lambda kv: kv[1]).cache()

    # get the target node to find path to
    # [ (index, name) ]
    end = names.filter(lambda kv: kv[1] == target).cache()

    i = 1
    previous_sum_distance = 0
    while (True):
        begin_sum_distance = sum_distance.value

        assert copartitioned(root, edges)

        # [ ([page_ids], (distance, 'path')) ]
        neighbors = edges.join(root).values().flatMap(
            lambda kv: l_add_distance(kv, i))

        distinct_neighbors = neighbors.reduceByKey(
            lambda v1, v2: v1, numPartitions=edges.getNumPartitions())

        assert copartitioned(distinct_neighbors, nodes)
        nodes = distinct_neighbors.union(nodes).reduceByKey(
            l_min_distance).cache()

        assert copartitioned(end, nodes)
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

        root = distinct_neighbors.cache()
        i += 1
        previous_sum_distance = sum_distance.value - begin_sum_distance

    assert copartitioned(end, nodes)
    root_end_path = end.join(nodes).cache()

    return nodes, edges, root_end_path


def bfs(links_file, titles_file, root, target):
    start = time.time()

    nodes, edges, names = create_graph(links_file, titles_file)
    _, _, root_end_path = bfs_search(nodes, edges, names, root, target)

    # get the path containing page ids
    # [ (id, 0) ]
    path_page_ids = root_end_path.values().flatMap(
        lambda kv: kv[1][1].split()).map(
        lambda v: (int(v), 0)).partitionBy(names.getNumPartitions())

    assert copartitioned(path_page_ids, names)
    path = path_page_ids.join(names).map(lambda kv: kv[1][1]).collect()

    print '-------- FOUND PATH --------'
    print path

    print 'TIME ' + repr(time.time() - start)

quiet_logs(sc)

# bfs('links.smp', 'titles.smp', 'stress', 'harvard')

bfs('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt',
    's3://Harvard-CS205/wikipedia/titles-sorted.txt',
    'Kevin_Bacon',
    'Harvard_University')

bfs('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt',
    's3://Harvard-CS205/wikipedia/titles-sorted.txt',
    'Harvard_University',
    'Kevin_Bacon')
