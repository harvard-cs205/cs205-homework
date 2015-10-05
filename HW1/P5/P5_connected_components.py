from operator import add
import time
import sys

import findspark
findspark.init('/home/lhoang/spark')

import pyspark
sc = pyspark.SparkContext(appName="spark1")

partition_size = 32
default_distance = sys.maxint
sum_distance = sc.accumulator(0)
sum_neighbors = sc.accumulator(0)


def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner


def quiet_logs(sc):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.WARN)
    logger.LogManager.getLogger("amazonaws").setLevel(logger.Level.WARN)


def l_get_symmetric_pairs(edge):
    '''
    Expand an edge from compressed representation to symmetric representation
    mapping from each node to its neighbor.
    :param edge: The edge represented as (node, [neighbors(node)])
    '''
    neighbors = edge[1].split()
    for nb in neighbors:
        yield (int(edge[0]), [int(nb)])
        yield (int(nb), [int(edge[0])])


def aggregate_distance(d):
    global sum_distance
    if d < default_distance:
        sum_distance += 1


def aggregate_neighbor_count(neighbor_list):
    global sum_neighbors
    sum_neighbors += len(neighbor_list)


def aggregate_neighbor_id(neighbor_id):
    global sum_neighbors
    sum_neighbors += neighbor_id


def flat_merge_list(id_list1, id_list2):
    return list(set(id_list1 + id_list2))


def merge_tuple_kv(list_kv):
    return list(set(list_kv[0] + list_kv[1]))


def create_symmetric_graph(fileName):
    """
    Create a symmetric graph of wiki pages.
    :param fileName: The file to load.
    """
    src = sc.textFile(fileName)

    # make symmetric links then reduce to give a list of
    # (page_i, [neighbors of page_i])
    edges = src.map(lambda x: x.split(':')).flatMap(
        l_get_symmetric_pairs).reduceByKey(
        add, numPartitions=partition_size).mapValues(
        lambda v: list(set(v))).cache()

    # create list of nodes with initial distance
    # no need to cached here as it will be done later after another
    # transformation before iterative loop
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

    # get edge pairs [(page_i, page_j)]
    edge_pairs = src.map(lambda x: x.split(':')).flatMap(
        lambda kv: [(int(kv[0]), int(nb)) for nb in kv[1].split()])

    # get reverse edge pairs [(page_j, page_i)]
    edge_pairs_reverse = edge_pairs.map(lambda kv: (kv[1], kv[0]))

    # intersect edge and reverse_edge to keep only the dual links
    edges = edge_pairs.intersection(edge_pairs_reverse).mapValues(
        lambda v: [v]).reduceByKey(
        add, numPartitions=partition_size).mapValues(
        lambda v: list(set(v))).cache()

    # create list of nodes with initial distance
    # no need to cached here as it will be done later after another
    # transformation before iterative loop
    nodes = edges.mapValues(lambda v: default_distance).cache()

    return nodes, edges


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
        nodes.foreach(lambda kv: aggregate_distance(kv[1]))
        if sum_distance.value - begin_sum_distance == previous_sum_distance:
            print 'Converged after ' + repr(i) + ' iterations'
            break

        root = distinct_neighbors.cache()
        i += 1
        previous_sum_distance = sum_distance.value - begin_sum_distance

    return nodes, edges


def log_diameter_search(edges):
    '''
    Algorithm for computing connected components in log(D) iterations.
    This works by sending merging all neighbors of one node with all
    neighbors of its neighbors. For example, if 1 is connected to 2 & 3
    then [2, 3] is merged with neighbors of 2 and neighbors of 3.
    '''
    global sum_neighbors

    i = 0
    previous_sum_neighbors = 0
    while True:
        sum_neighbors += -sum_neighbors.value

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

        print '-i ' + repr(i) + ', sum: ' + repr(previous_sum_neighbors)

        # check for convergence using accumulator
        merged_edges.foreach(lambda kv: aggregate_neighbor_count(kv[1]))
        if sum_neighbors.value == previous_sum_neighbors:
            print 'Converged after ' + repr(i) + ' iterations'
            break

        print '+i ' + repr(i) + ', sum: ' + repr(previous_sum_neighbors)
        edges = merged_edges.cache()
        i += 1
        previous_sum_neighbors = sum_neighbors.value

    return edges


def min_search(nodes, edges):
    '''
    The intuition is as follows: each node belongs to a certain connected
    component. We can label each component by the node with the minimum
    id and want to set the value of each node in the component to this id.
    At each iteration, every node passes its value to its neighbors and
    the neighbors keep the minimum of all values received. Eventually
    the true label of the component (i.e. the minimum id of its member nodes)
    will propagate to every node that belongs to it. This way we also know
    the membership of each node, i.e. which component it belongs to.
    '''
    global sum_neighbors

    # initially set the value of each node to itself
    node_membership = nodes.map(
        lambda kv: (kv[0], kv[0]), preservesPartitioning=True).cache()

    i = 0
    previous_sum_neighbors = 0
    while True:
        print 'ITERATION ' + repr(i)

        # reset accumulator
        sum_neighbors += -sum_neighbors.value

        assert copartitioned(node_membership, edges)
        # propagate the value of each node to its neighbors
        new_node_membership = node_membership.join(edges).values(
            ).flatMap(lambda kv: [(nid, kv[0]) for nid in kv[1]])

        # each neighbor computes the minimum id from those it received
        new_node_membership_reduced = new_node_membership.reduceByKey(
            min, numPartitions=nodes.getNumPartitions())

        assert copartitioned(node_membership, new_node_membership_reduced)

        # each neighbor takes the new id if it's smaller the its current value
        final_node_membership = new_node_membership_reduced.union(
            node_membership).reduceByKey(min)

        # check for convergence using accumulator
        final_node_membership.foreach(lambda kv: aggregate_neighbor_id(kv[1]))

        if sum_neighbors.value == previous_sum_neighbors:
            print 'Converged after ' + repr(i) + ' iterations'
            break

        node_membership = final_node_membership.cache()

        previous_sum_neighbors = sum_neighbors.value
        i += 1

    return node_membership


def find_components_bfs_search(nodes, edges):
    '''
    Find connected components by performing bfs search iteratively.
    At each iteration, the algorithm finds a node that has not been
    visited, use it as the source, and perform bfs search to visit all
    of its neighbors.
    '''
    root = nodes.first()[0]

    global sum_distance

    num_components = 0
    num_total_elements = nodes.count()
    num_total_component_elements = 0

    visited_nodes = nodes
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

        assert copartitioned(visited_nodes, nodes_bfs)

        # combine with previous bfs graph to update the list
        # of visited nodes
        new_visited_nodes = visited_nodes.union(
            nodes_bfs).reduceByKey(min)

        # if all nodes are touched then quit
        untouched_nodes = new_visited_nodes.filter(
            lambda kv: kv[1] == default_distance)
        if untouched_nodes.isEmpty():
            break

        # set visited nodes for next iteration
        visited_nodes = new_visited_nodes.cache()

        # get the first node that hasn't been touched to search again
        root = untouched_nodes.first()[0]


def find_components_log_diameter(nodes, edges):
    edges_components = log_diameter_search(edges)

    components = edges_components.map(
        lambda kv: ' '.join(sorted(kv[1]))).distinct()

    max_num_elements = components.map(
        lambda v: len(v.split())).takeOrdered(1, lambda v: -v)

    print '# of components: ' + repr(components.count())
    print 'max # of elements: ' + repr(max_num_elements)


def find_components_min(nodes, edges):
    print 'FINDING COMPONENTS USING MIN_SEARCH'

    nodes_min = min_search(nodes, edges)
    components = nodes_min.map(lambda kv: (kv[1], [kv[0]])).reduceByKey(
        add).mapValues(lambda neighbor_list: len(neighbor_list))

    print 'Most number of elements:'
    print components.takeOrdered(5, lambda kv: -kv[1])

    print 'Number of components:'
    print components.count()


def main():
    quiet_logs(sc)

    # links_file = 'links.smp'
    # links_file = 'links_bug.smp'
    # links_file = '../P4/source.csv'
    # links_file = 'links-simple-sorted-10k.txt'
    links_file = 'links-simple-sorted.txt'
    # links_file = 's3://Harvard-CS205/wikipedia/links-simple-sorted.txt'

    # print '----- Symmetric Graph -----'
    # start = time.time()

    # nodes_sym, edges_sym = create_symmetric_graph(links_file)
    # find_components_min(nodes_sym, edges_sym)

    # print 'Time: ' + repr(time.time() - start)

    # print '---------------------------'

    print '-----   Dual Graph    -----'
    start = time.time()

    nodes_dual, edges_dual = create_dual_graph(links_file)
    find_components_min(nodes_dual, edges_dual)

    print 'Time: ' + repr(time.time() - start)

    print 'FINISHED'

main()
