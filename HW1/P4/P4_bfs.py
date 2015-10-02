
# coding: utf-8

# In[1]:

# Version assuming graph diameter = 10
def do_bfs1(sc, source_node, hero_graph):
    n_parts = 20
    curr_nodes = sc.parallelize([(source_node, 0)], n_parts)
    curr_nodes = curr_nodes.partitionBy(n_parts, hash)

    graph_diam = 10
    for iter_i in range(graph_diam):
        neighbors = (curr_nodes.join(hero_graph)
                     .flatMap(lambda x: x[1][1])
                     .map(lambda x: (x, iter_i + 1)))
        new_nodes = neighbors.subtractByKey(curr_nodes, n_parts).cache()
        curr_nodes = ((curr_nodes + new_nodes)
                      .repartition(n_parts).cache())
    return curr_nodes


# In[2]:

# Version without assuming graph diameter, using accumulator
def do_bfs2(sc, source_node, hero_graph):
    n_parts = 20
    curr_nodes = sc.parallelize([(source_node, 0)], n_parts)
    curr_nodes = curr_nodes.partitionBy(n_parts, hash)

    accum = sc.accumulator(0)
    iter_i = 0
    while accum.value == 0:
        neighbors = (curr_nodes.join(hero_graph)
                     .flatMap(lambda x: x[1][1])
                     .map(lambda x: (x, iter_i + 1)))
        new_nodes = neighbors.subtractByKey(curr_nodes, n_parts).cache()
        if new_nodes.count() == 0:
            accum.add(1)
        else:
            curr_nodes = ((curr_nodes + new_nodes)
                          .repartition(n_parts).cache())
            iter_i = iter_i + 1
    return curr_nodes

