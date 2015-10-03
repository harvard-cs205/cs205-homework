from operator import add
import time
import sys

# import findspark
# findspark.init('/home/lhoang/spark')

import pyspark
sc = pyspark.SparkContext(appName="spark1")

partition_size = 20
default_distance = sys.maxint
sum_distance = sc.accumulator(0)
sum_neighbors = sc.accumulator(0)

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

    # create list of nodes with initial distance
    # no need to cached here as it will be done later after another
    # transformation before iterative loop
    nodes = edges.mapValues(lambda v: default_distance)

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

    # create list of nodes with initial distance
    # no need to cached here as it will be done later after another
    # transformation before iterative loop
    nodes = edges.mapValues(lambda v: default_distance)

    return nodes, edges


def l_add_distance(list, d):
    for kv in list:
        yield (kv, d)


def update_accumulator(d):
    global sum_distance
    if d < default_distance:
        sum_distance += 1


def tally_neighbors(neighbor_list):
    global sum_neighbors
    sum_neighbors += len(neighbor_list)
    pass


def flat_merge_list(id_list1, id_list2):
    return list(set(id_list1 + id_list2))


def merge_tuple_kv(list_kv):
    return list(set(list_kv[0] + list_kv[1]))


def bfs_search(nodes, edges, page):
    """
    Perform breadth-first search to find shortest path from root.
    :param page: The root page to find paths from.
    """
    root = nodes.filter(lambda kv: kv[0] == page)

    # set the root node to distance 1 to make it a touched node
    # to make sure it's not used for the next connected component search
    nodes = nodes.map(lambda kv: (kv[0], 1 if kv[
        0] == page else kv[1]), preservesPartitioning=True).cache()

    assert copartitioned(nodes, edges)

    i = 1
    previous_sum_distance = 0
    while (True):
        begin_sum_distance = sum_distance.value

        assert copartitioned(root, edges)

        # get neighbors of root
        # [ (id, distance) ]
        neighbors = edges.join(root).values().flatMap(
            lambda kv: [(v, i) for v in kv[0]])

        # get distinct neighbors and copartition with edges
        distinct_neighbors = neighbors.reduceByKey(
            lambda v1, v2: v1, numPartitions=edges.getNumPartitions())

        assert copartitioned(distinct_neighbors, nodes)

        # update distances in nodes by reducing with min on values
        # format of nodes is [ (id, distance) ]
        nodes = distinct_neighbors.union(nodes).reduceByKey(min).cache()

        # check for convergence using accumulator
        nodes.foreach(lambda kv: update_accumulator(kv[1]))
        if sum_distance.value - begin_sum_distance == previous_sum_distance:
            print 'Converged after ' + repr(i) + ' iterations'
            break

        root = distinct_neighbors.cache()
        i += 1
        previous_sum_distance = sum_distance.value - begin_sum_distance

    return nodes, edges


def log_diameter_search(edges):

    i = 0
    previous_sum_neighbors = 0
    while True:
        begin_sum_neighbors = sum_neighbors.value

        # take the neighbors [page_1, page_2, ...] and maps to
        # [ (page_1, [page_1, page_2, ...]) ]
        neighbor_edges = edges.values().flatMap(
            lambda id_list: [(id, id_list) for id in id_list])

        # reduce so that each page id maps to a unique list of
        # connected neighbors
        # [ (page_1, [page_1, page_2, ...]) ]
        distinct_neighbor_edges = neighbor_edges.reduceByKey(
            flat_merge_list, numPartitions=edges.getNumPartitions())

        assert copartitioned(distinct_neighbor_edges, edges)

        # join with original edge set to merge with new neighbors
        # so that the set of immediate neighbors grows to the set of
        # connected neighbors
        # [ (page_1, [page_1, page_2, ...]) ]
        merged_edges = distinct_neighbor_edges.join(edges).mapValues(
            merge_tuple_kv)

        # check for convergence using accumulator
        merged_edges.foreach(lambda kv: tally_neighbors(kv[1]))
        if sum_neighbors.value - begin_sum_neighbors == previous_sum_neighbors:
            print 'Converged after ' + repr(i) + ' iterations'
            break

        edges = merged_edges.cache()
        i += 1
        previous_sum_neighbors = sum_neighbors.value - begin_sum_neighbors

    return edges


def find_components(nodes, edges):
    root = nodes.first()[0]

    global sum_distance

    num_components = 0
    num_total_elements = nodes.count()
    num_total_component_elements = 0
    while True:
        sum_distance += -sum_distance.value
        nodes_bfs, _ = bfs_search(nodes, edges, root)

        num_components += 1

        print 'COMPONENT ' + repr(num_components)
        print 'root = ' + repr(root)

        num_elements = nodes_bfs.filter(
            lambda kv: kv[1] < default_distance).count()
        print '# of connected elements = ' + repr(num_elements)

        # if sum of elements in all components reaches total then quit
        num_total_component_elements += num_elements
        if num_total_component_elements >= num_total_elements:
            break

        # if all nodes are touched then quit
        untouched_nodes = nodes_bfs.filter(
            lambda kv: kv[1] == default_distance)
        if untouched_nodes.isEmpty():
            break

        # get the first node that hasn't been touched to search again
        root = untouched_nodes.first()[0]


def find_components_log_diameter(nodes, edges):
    edges_components = log_diameter_search(edges)
    components = edges_components.map(
        lambda kv: ' '.join(sorted(kv[1]))).distinct().collect()
    print components


quiet_logs(sc)

# links_file = 'links.smp'
links_file = 's3://Harvard-CS205/wikipedia/links-simple-sorted.txt'

print '----- Symmetric Graph -----'
start = time.time()

nodes_sym, edges_sym = create_symmetric_graph(links_file)
find_components_log_diameter(nodes_sym, edges_sym)

print 'Time: ' + repr(time.time() - start)

print '---------------------------'

print '-----   Dual Graph    -----'
start = time.time()

nodes_dual, edges_dual = create_dual_graph(links_file)
find_components_log_diameter(nodes_dual, edges_dual)

print 'Time: ' + repr(time.time() - start)

print 'FINISHED'
