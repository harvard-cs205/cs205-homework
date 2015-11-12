

# Version 3
# SS-BFS Algorithm: only one rdd without collecting/diameter
# TODO: Shuffle optimization done: groupByKey() ==> reduceByKey() and mapValues
def ss_bfs3(bfs_graph, root, partitions=4):
    '''
    Return a (K, (d, [neighbors])) rdd with the nodes as keys and d distance
    from the root as value (d = inf is not connected to the root) and its
    neighbors list in the graph [neighbors].
    Entry graph should be formated as follows:
    (K, (d, [neighbors])), with eventually d = inf for all the nodes if first
    call to ss_bfs3
    '''
    # WARNING: the value need to be a tuple
    current = bfs_graph.filter(lambda x: x[0] == root)\
        .mapValues(lambda v: (0, v[1]))
    i = 0
    while True:
        # Visiting the neighbors of the Nodes in the subgraph current and
        # updating their distance.
        # Redundancy may happen, hence the use of distinct()
        visiting = current.flatMap(lambda x: [(n, (x[1][0] + 1, 1))
                                   for n in list(x[1][1])]).distinct()\
            .partitionBy(partitions)
        # Updating the current graph
        bfs_graph = bfs_graph.union(visiting).groupByKey()\
            .mapValues(lambda v: (min([n[0] for n in list(v)]),
                       list(v)[0][1]))
        # Need to update the indices before using it in spark (cf. lazzy)
        i += 1
        # Getting the actual visiting nodes (not previously visited)
        current = bfs_graph.filter(lambda x: x[1][0] == i)
        # Check that nodes are left
        if not current.count():
            break
    return bfs_graph


# Version 2
# SS-BFS Algorithm: without collecting, create a second rdd
def ss_bfs2(graph, root, partitions=4):
    '''
    Return a (K, V) rdd with the visited nodes as keys and their distance from
    the root as value.
    Computation is slow because of the collect() call at each stage.
    '''
    # Initialization
    # WARNING: the value need to be a tuple
    current = graph.filter(lambda x: x[0] == root)\
        .mapValues(lambda v: (0, v))
    distance = graph.context.parallelize([(root, 0)])
    i = 0
    while True:
        # Visiting the neighbors of the Nodes in the subgraph current and
        # updating their distance.
        # Redundancy may happen, hence the use of distinct()
        visiting = current.flatMap(lambda x: [(n, x[1][0] + 1)
                                   for n in list(x[1][1])]).distinct()\
            .partitionBy(partitions)
        distance = visiting.union(distance).groupByKey()\
            .mapValues(lambda v: min([d for d in list(v)]))\
            .partitionBy(partitions)
        # Getting the actual visiting nodes (not previously visited)
        visiting_ = distance.filter(lambda x: x[1] == i + 1)
        # Check that nodes are left
        if not visiting_.count():
            break
        # Getting the list of the neighbors of the current nodes as values.
        current = visiting_.join(graph)
        i += 1
    return distance
