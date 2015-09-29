def BFS(graph, source_node, max_iters):
    """ Runs a parallel BFS on graph.
    Graph is assumed to in the adjacency list representation of (node, (distance, [neighbors])).
    Distances are initilized to infinite in this function for assurance.
    Returns a graph RDD of the same form, only with correct distances.
    """
    # First initialize all of the distances
    dist_graph = graph.map(lambda (x, y): (x, (0, y)) if (x == source_node) else (x, (10**8, y)))
    
    # Now map over the group, exploring one level each time
    for ii in range(max_iters):

        # Explore one level from each node
        dist_graph = dist_graph.flatMap(explore_level_from_node)


        # Combine results to get shortest distances and reconstruct adjacency lists
        dist_graph = dist_graph.reduceByKey(find_actual_distances)

    return dist_graph

def explore_level_from_node(node):
    """ Explores one level from the node node.
    Node is assumed to come in the form (node, (current_best_distance_to_source), [neighbors]).
    Updates each node connected to this node to have distance +1, and returns a list containing
    the parent node that we came from.
    To be used with flatMap to parallel BFS from each node. Some of the returned distances
    will NOT be optimal - these will be eliminated with find_actual_distances.
    """

    # Just split up the data for readability
    curr_node = node[0]
    d = node[1][0]
    adj_list = node[1][1]

    # Our list of results to be cast into a tuple and returned for flatMap
    results = []

    # Iterate over all other nodes, update their distances, and return the parent.
    if (d < 10**8):
        for other_node in adj_list:
            # Append our tentative new distance
            results.append((other_node, (d+1, [curr_node])))

    # And lets not lose any nodes now...
    results.append(node)

    return tuple(results)

def find_actual_distances(d1_par1, d2_par2):
    """ Reduce function for a graph RDD of the form (node, (distance_to_source, [parents])).
    We assume the graph RDD just came out of explore_level_from_node, and thus has many duplicate
    nodes and distances - most of which are not optimal. We reduce by taking only the shortest distance
    from the source, as this is the distance we are interested in. We also combine ALL parents
    to reconstruct the list of neighbors.
    """
    # Get the first distance and parents
    d1 = d1_par1[0]
    par1 = d1_par1[1]

    # Get the second distance and parents
    d2 = d2_par2[0]
    par2 = d2_par2[1]

    # Find the actual distance
    d = 0
    if d1 < d2:
        d = d1 
    else:
        d = d2

    # Reconstruct the adjacency list by combining parents
    # Remove any duplicates that may appear
    pars = list(set(par1 + par2))

    return (d, pars)
