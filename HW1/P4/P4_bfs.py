def update_dist(level, neighbors_at_level):
    '''
    Update Distances of all nodes given a current level and neighbors at next level
    '''
    return lambda (node, (dist, neighbors)): (node, (min(dist,level+1), neighbors)) if node in neighbors_at_level else (node, (dist, neighbors))

def BFS(rdd_graph, root_node, sc):
    '''
    Given a (K, V) RDD with K = node and V = [connecting nodes],
    find the shortest paths from a source node to all other nodes in a graph.
    '''
    # rdd_graph => (node, (distance, neighbors)) RDD
    rdd_graph = rdd_graph.map(lambda (node, neighbors): (node, (0, neighbors)) if (node == root_node) else (node, (float('inf'), neighbors)))
    accum = sc.accumulator(0)
    level = 0
    while True:
      # Store nodes at a current level into queue
      rdd_queue = rdd_graph.filter(lambda (node, (dist, neighbors)) : dist == level)
      # if queue is empty -> break
      if rdd_queue.count() == 0:
        break
      else:
        # Create a list of neighbors of nodes at a current level
        rdd_queue = rdd_queue.map(lambda (node, (dist, neighbors)): neighbors)
        neighbors_at_level = rdd_queue.reduce(lambda x,y: list(set(x+y)))
        # Update distances of the neighbors 
        rdd_graph = rdd_graph.map(update_dist(level, neighbors_at_level))
        # Go to next level
        accum.add(1)
        level += 1
    
    # Reform rdd into (node, distance)
    rdd_result = rdd_graph.filter(lambda (node, (dist, neighbors)): dist != float('inf'))
    rdd_result = rdd_result.map(lambda (node, (dist, neighbors)): (node, dist))
    rdd_result = rdd_result.sortBy(lambda (node, dist): dist, ascending=False)   
    result = rdd_result.collect()

    numberAll = rdd_graph.count()
    print "Root Node : %s" % root_node
    print "The number of touched nodes : %i" % (len(result)-1)
    print "The number of untouched nodes : %i" % (numberAll - (len(result)-1))
    print "Diameter : %i" % accum.value

    return result
