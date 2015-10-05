
# coding: utf-8

# In[ ]:

def bfs(sc, adj_matrix_rdd, char, diameter=float("inf")):
    '''
    Takes SparkContext (sc), adjacency matrix rdd (adj_matrix_rdd), character name (char), and optional diameter
    of graph as parameter and performs BFS.
    Note: adj_matrix_rdd contain 
        key = character name
        value = list of names of adjacent characters                   
    '''
    def group_neighbor_dist(neighbors,dist):
        return [(n,dist) for n in neighbors]
    
    dist = 0
    dists_rdd = sc.parallelize([(char,dist)]).partitionBy(100) #shortest paths to nodes
    neighbors_rdd = dists_rdd
    num_touched = 1
    #We assume graph diameter is <=10
    i = 0
    num_new_neighbors = 1 
    while (i < diameter) and (num_new_neighbors != 0):
        #get neighbors of neighbors
        neighbors_rdd = neighbors_rdd.join(adj_matrix_rdd) 
        #we only care about the new neighbors
        neighbors_rdd = neighbors_rdd.values()            .flatMap(lambda (prev_dist,neighbors): group_neighbor_dist(neighbors,prev_dist+1))            .distinct()            .partitionBy(100)
            
        #we now have rdd of (char,dist)
        assert neighbors_rdd.partitioner == dists_rdd.partitioner, "neighbors and dists are not copartitioned"
        #remove characters that we already have a shorter path to
        neighbors_rdd = neighbors_rdd.subtractByKey(dists_rdd) 
        neighbors_rdd.cache()
        num_new_neighbors = neighbors_rdd.count()
        num_touched += num_new_neighbors
        #update dists_rdd to include the new nodes we have explored
        dists_rdd = dists_rdd.union(neighbors_rdd)
        i += 1
        
    print num_touched
    return dists_rdd

