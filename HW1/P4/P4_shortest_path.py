
# coding: utf-8

# In[1]:

def shortest_path(sc, adj_matrix_rdd, start_name, end_name, diameter=float("inf")):
    '''
    Takes SparkContext (sc), adjacency matrix rdd (adj_matrix_rdd), character name (char), and optional diameter
    of graph as parameter and performs BFS.
    Note: adj_matrix_rdd contains 
        key = character name
        value = list of names of adjacent characters
    Note: adj_matrix_rdd should be hash partitioned into 100 parts
    Returns (path length, list of nodes in path (including start))
    '''

    def make_neighbor_key(neighbors,prev_dist, path_to_current):
        #helper function for transforming values in RDDs
        return [(n, (prev_dist +1, path_to_current+[n])) for n in neighbors]
    
    dist = 0
    #paths_rdd with hold:
    #    key = name at end of path
    #    value = (path length, path) where path = list of nodes in path in order 
    #                               (i.e. [1st node name in path after start, second node name in path, ..., last node name in path])
    paths_rdd = sc.parallelize([(start_name,(dist, [start_name]))]).partitionBy(100) 
    neighbors_rdd = paths_rdd
    num_touched = 1
    #We assume graph diameter is <=10
    i = 0
    num_new_neighbors = 1 
    while (i < diameter) and (num_new_neighbors != 0):
        dist += 1
        #get neighbors of neighbors
        neighbors_rdd = neighbors_rdd.join(adj_matrix_rdd) 
        
        #we have (current node, (prev_dist, path_to_current_node, [neighbor1, neighbor2, ...])
        # we want (neighbor 1, (path_length, path_to neighbor1))
        #         ...
        # -->new_neighbors_rdd
        #eliminate duplicates, we only need one path to a node
        #print neighbors_rdd.values().take(1)
        neighbors_rdd = neighbors_rdd.values()            .flatMap(lambda ((prev_dist, path_to_current), neighbors): make_neighbor_key(neighbors,prev_dist,path_to_current))            .reduceByKey(lambda _, val: val)            .partitionBy(100)
        #we now have rdd of (char,dist)
        assert neighbors_rdd.partitioner == paths_rdd.partitioner, "neighbors and dists are not copartitioned"
        #get only unexplored nodes / remove nodes that we already have a shorter path to
        neighbors_rdd = neighbors_rdd.subtractByKey(paths_rdd)
        neighbors_rdd.cache()
        num_new_neighbors = neighbors_rdd.count()
        num_touched += num_new_neighbors
        #update dists_rdd to include the new nodes we have explored
        paths_rdd = paths_rdd.union(neighbors_rdd)
        i += 1
        path_to_end = paths_rdd.lookup(end_name)
        if path_to_end != []:
            return path_to_end[0]
    print num_touched
    return None


# In[ ]:



