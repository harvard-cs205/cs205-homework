
# coding: utf-8

# In[ ]:

def bfs(sc, adj_matrix_rdd, char, diameter=float("inf")):
    '''
    Takes SparkContext (sc), adjacency matrix rdd (adj_matrix_rdd), character name (char), and optional diameter
    of graph as parameter and performs BFS.
    Note: adj_matrix_rdd contains 
        key = character name
        value = list of names of adjacent characters
    Note: adj_matrix_rdd should be hash partitioned into 100 parts
    '''
    def group_neighbor_dist(neighbors,dist):
        return [(n,dist) for n in neighbors]
    dist = 0
    
    num_iter_accum = sc.accumulator(0)
    dists_rdd = sc.parallelize([(char,dist)]).partitionBy(100) #shortest paths to nodes
    neighbors_rdd = dists_rdd
    #We assume graph diameter is <=10
    num_new_neighbors_accum = sc.accumulator(1)
    while (num_iter_accum.value < diameter) and (num_new_neighbors_accum.value != 0):
        #get neighbors of neighbors
        num_new_neighbors_accum = sc.accumulator(0) # reset number of new neighbors for this iteration
        neighbors_rdd = neighbors_rdd.join(adj_matrix_rdd) 
        #we only care about the new neighbors
        neighbors_rdd = neighbors_rdd.values()            .flatMap(lambda (prev_dist,neighbors): group_neighbor_dist(neighbors,prev_dist+1))            .distinct()            .partitionBy(100)
        #we now have rdd of (char,dist)
        assert neighbors_rdd.partitioner == dists_rdd.partitioner, "neighbors and dists are not copartitioned"
        #remove characters that we already have a shorter path to
        neighbors_rdd = neighbors_rdd.subtractByKey(dists_rdd) 
        neighbors_rdd.cache()
        #update dists_rdd to include the new nodes we have explored
        dists_rdd = dists_rdd.union(neighbors_rdd)
        neighbors_rdd.foreach(lambda _ :num_new_neighbors_accum.add(1))
        num_iter_accum.add(1)
    
    print dists_rdd.count() #num_touched
    return dists_rdd

