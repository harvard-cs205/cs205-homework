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


def connected_components(graph):
    '''
    Computes the number of connected components in the graph.
    Return (N, top) with N the number of connected components and top the
    number of nodes in the largest connected component.
    '''
    # Initialization of the graph: formatting it to call ss_bfs3
    current = graph.mapValues(lambda v: (float('inf'), v))
    # Initialization of the return variables
    N = 0
    top = 0
    while current.count():
        root = current.keys().take(1)[0]
        # Debugg
        # print('current root is {}'.format(root))
        # Computing next connected component
        graph_with_component = ss_bfs3(current, root)
        # Updating return
        m = graph_with_component.filter(lambda x: x[1][0] < float('inf'))\
            .count()
        N += 1
        top = max(top, m)
        # Debugg
        # print('current component count is {}'.format(m))
        # Filtering the visited nodes: filter return false for the node x in
        # the component subgraph
        # Call cache() to persist this stage in cache
        current = graph_with_component\
            .filter(lambda x: x[1][0] == float('inf')).cache()
        # Debugg
        # print('current subgraph count is {}'.format(current.count()))
    return (N, top)
