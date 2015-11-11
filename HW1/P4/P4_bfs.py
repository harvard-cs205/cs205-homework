
# coding: utf-8

# In[6]:

def make_hero_graph(text_path, sc, n_parts):

    # Load marvel comic data
    dat = sc.textFile(text_path)

    # Remove quotations and split issue & hero name
    dat_split = dat.map(lambda x: x[1:-1].split('","'))

    # Mapping between issue->hero
    dat_comic = dat_split.map(lambda x: (x[1], x[0]))
    dat_comic = dat_comic.partitionBy(n_parts).cache()

    comic_key = dat_comic.combineByKey(lambda x: {x}, 
                                       lambda a, b: a.union({b}), 
                                       lambda a, b: a.union(b))
    assert dat_comic.partitioner == comic_key.partitioner

    comic_hero_join = dat_comic.join(comic_key).map(lambda x: x[1])
    comic_hero_join = comic_hero_join.partitionBy(n_parts)

    hero_graph = comic_hero_join.combineByKey(lambda x: x, 
                                     lambda a, b: a.union(b), 
                                     lambda a, b: a.union(b))
    hero_graph = hero_graph.map(lambda KV: (KV[0], KV[1] - {KV[0]}), True)
    assert hero_graph.partitioner == comic_key.partitioner
    return hero_graph


# In[1]:

# Version assuming graph diameter = 10
def do_bfs1(sc, source_node, hero_graph, n_parts):
    curr_nodes = sc.parallelize([(source_node, 0)], n_parts)

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
def do_bfs2(sc, source_node, hero_graph, n_parts, stop_node):
    # Make sure pre-join RDDs are copartitioned
    node_hist = (sc.parallelize([(source_node, 0)])
                 .partitionBy(n_parts, hash))
    new_nodes = node_hist
    hero_graph = hero_graph.partitionBy(n_parts, hash).cache()
    hero_filt = hero_graph
    assert new_nodes.partitioner == hero_graph.partitioner

    # Keep track of whether there are no new nodes touched
    accum = sc.accumulator(1)

    # Distance corresponding to current iteration
    iter_i = 0 

    while (accum.value > 0 
           and new_nodes.filter(lambda x: x[0] == stop_node).count() == 0):
        print('Starting iteration {}'.format(iter_i))
        assert new_nodes.partitioner == hero_filt.partitioner
        #         def count_map(K, accum):
        #             accum.add(1)
        #             return (K, iter_i + 1)
        neighbors = (new_nodes.join(hero_filt)
                     .flatMap(lambda x: x[1][1])
                     .distinct()
                     .map(lambda x: (x, iter_i + 1)))

        hero_filt = hero_filt.subtractByKey(new_nodes)
        hero_filt = hero_filt.partitionBy(n_parts, hash).cache()

        # Take away the nodes that were already explored; these are not new
        new_nodes = neighbors.subtractByKey(node_hist)
        new_nodes = new_nodes.partitionBy(n_parts, hash).cache()

        # Use an accumulator for no reason; 
        # equivalent to performing a count on new_nodes
        accum = sc.accumulator(0)
        new_nodes.foreach(lambda _: accum.add(1))
        node_hist = (node_hist + new_nodes).cache()



        iter_i = iter_i + 1
    return node_hist


# In[ ]:



