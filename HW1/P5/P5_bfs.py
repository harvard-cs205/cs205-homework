
# coding: utf-8

# In[1]:

import pyspark


# In[2]:

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
    num_iter_accum = sc.accumulator(0)
    num_new_accum = sc.accumulator(1)
    #We assume graph diameter is <=10
    while (num_iter_accum.value < diameter) and (num_new_accum.value != 0):
        dist += 1
        #get neighbors of neighbors
        num_new_accum = sc.accumulator(0)
        neighbors_rdd = neighbors_rdd.join(adj_matrix_rdd) 
        
        #we have (current node, (prev_dist, path_to_current_node, [neighbor1, neighbor2, ...])
        # we want (neighbor 1, (path_length, path_to neighbor1))
        #         ...
        # -->new_neighbors_rdd
        #eliminate duplicates, we only need one path to a node
        neighbors_rdd = neighbors_rdd.values()            .flatMap(lambda ((prev_dist, path_to_current), neighbors): make_neighbor_key(neighbors,prev_dist,path_to_current))            .reduceByKey(lambda _, val: val)            .partitionBy(100)
        assert neighbors_rdd.partitioner == paths_rdd.partitioner, "neighbors and dists are not copartitioned"
        #get only unexplored nodes / remove nodes that we already have a shorter path to
        neighbors_rdd = neighbors_rdd.subtractByKey(paths_rdd)
        neighbors_rdd.cache()
        neighbors_rdd.foreach(lambda _ : num_new_accum.add(1))
        #update dists_rdd to include the new nodes we have explored
        paths_rdd = paths_rdd.union(neighbors_rdd)
        num_iter_accum.add(1)
        path_to_end = paths_rdd.lookup(end_name)
        if path_to_end != []:
            print paths_rdd.count()
            return path_to_end[0]
    print paths_rdd.count()
    return None


# In[4]:

#NOTE: Links is 10x larger than Pages --> Partition using this
#NOTE: This is a DIRECTED GRAPH
#source_rdd will contain (key=character, value=a comic that the character is in)


#Adapted from https://github.com/thouis/SparkPageRank/blob/master/PageRank.py
def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) for to in dests.split(' ')]
    return (int(src), dests)

if __name__ == '__main__':
    sc = pyspark.SparkContext()
    sc.setLogLevel('WARN')

    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
    page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)

    # process links into (node #, [neighbor node #, neighbor node #, ...]
    neighbor_graph = links.map(link_string_to_KV)

    # create an RDD for looking up page names from numbers
    # remember that it's all 1-indexed
    page_names = page_names.zipWithIndex().map(lambda (n, id): (id + 1, n))
    page_names = page_names.sortByKey().cache()

    #######################################################################
    # set up partitioning - we have roughly 16 workers, if we're on AWS with 4
    # nodes not counting the driver.  This is 16 partitions per worker.
    #
    # Cache this result, so we don't recompute the link_string_to_KV() each time.
    #######################################################################
    neighbor_graph = neighbor_graph.partitionBy(256).cache()

    # find Kevin Bacon
    Kevin_Bacon = page_names.filter(lambda (K, V): V == 'Kevin_Bacon').collect()
    # This should be [(node_id, 'Kevin_Bacon')]
    assert len(Kevin_Bacon) == 1
    Kevin_Bacon = Kevin_Bacon[0][0]  # extract node id

    # find Harvard University
    Harvard_University = page_names.filter(lambda (K, V):
                                           V == 'Harvard_University').collect()
    # This should be [(node_id, 'Harvard_University')]
    assert len(Harvard_University) == 1
    Harvard_University = Harvard_University[0][0]  # extract node id

    #now shortest_path
    names = [Kevin_Bacon,Harvard_University]
    for i in range(2):
        #compute shortest path in each directipon
        dist,path = shortest_path(sc, neighbor_graph, names[i], names[i-1])
        named_path = [page_names.lookup(node)[0] for node in path]
        print dist, named_path


# In[ ]:




# In[ ]:



