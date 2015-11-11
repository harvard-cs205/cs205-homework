
# coding: utf-8

# In[ ]:

# Version without assuming graph diameter, using accumulator
def do_bfs_aws(sc, source_node, hero_graph, n_parts, stop_node):
    # Make sure pre-join RDDs are copartitioned
    node_hist = (sc.parallelize([(source_node, 0)]).partitionBy(n_parts, hash))
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
        print('Starting iteration ' + str(iter_i))
        print(str(new_nodes.count()) + 'new nodes touched')
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

def find_short_path(source_node, dest_node, hero_graph, n_parts, sc):
    node_hist = do_bfs_aws(sc, source_node, hero_graph, n_parts, dest_node)
    # Find what the shortest distance from source to destination is
    dest_dist = node_hist.lookup(dest_node)[0]
    dest_set = set([dest_node])

    shortest_path_options = [[dest_node]]
    for dist_i in range(dest_dist - 1, 0, -1):
        # Working backward from the destination, 
        # find all the nodes that are pointing to it
        connected_nodes = (hero_graph.filter(
            lambda x: len(dest_set.intersection(x[1])) > 0))

        # Find all the nodes along the shortest paths 
        # according to our BFS search 
        shortest_path_nodes = node_hist.filter(lambda x: x[1] == dist_i)

        # The intersection of these are the nodes that are potentially
        # along the shortest path between the source and destination nodes.
        possible_path_node = connected_nodes.join(shortest_path_nodes)
        dest_set = set(possible_path_node.keys().collect())

        shortest_path_options.append(list(dest_set))
    shortest_path_options.append([source_node])
    # Return one possible shortest path
    return [options[0] for options in reversed(shortest_path_options)]


# In[ ]:




# In[ ]:

# Function that finds the number of connected components in a graph
def find_connect_comp(sc, hero_graph, n_parts):
    import time
    # Get list of all heroes
    hero_list = hero_graph.keys().cache()
    curr_hero_graph = hero_graph
    connected_count = 0 # Count the number of connected components
    connect_hist = [] # Record information about connected components

    while hero_list.count() > 0:
        t1 = time.time()
        print(connected_count)

        # Take a node that hasn't been explored yet
        if connected_count == 0:
            source_node = hero_list.takeSample(False, 1)[0]
        else:
            source_node = hero_list.first()

        # Do BFS and recover all touched nodes
        print('Running BFS...')
        search_history = do_bfs_aws(sc, source_node, curr_hero_graph, n_parts, None)
        search_history = search_history.partitionBy(n_parts, hash).cache()

        # Determine remaining untouched nodes and repeat
        print('Pruning nodes...')
        hero_list = hero_list.subtract(search_history.keys(), n_parts).cache()

        # hero_list = hero_list.partitionBy(n_parts, hash).cache()

        # Remove nodes that have been explored 
        curr_hero_graph = curr_hero_graph.subtractByKey(search_history)
        curr_hero_graph = curr_hero_graph.partitionBy(n_parts, hash).cache()

        t2 = time.time()
        print((search_history.count(), curr_hero_graph.count(), t2 - t1))
        connect_hist.append((connected_count, search_history.count(), t2 - t1))
        connected_count = connected_count + 1
    return connect_hist


# In[ ]:

# Save list of strings to file
def save_str_list(test_strs, save_file):
    with open(save_file, 'a') as f:
        for word_i, word in enumerate(test_strs):
            f.write(word)
            if word_i < len(test_strs) - 1:
                f.write(' ')
            else:
                f.write('.\n\n')

