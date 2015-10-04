from functools import partial
import time

MAX_DIST = 10**8

def BFS(graph, source_node, sc):
    """ Runs a parallel BFS on graph.
    Graph is assumed to in the adjacency list representation of (node, (distance, [neighbors])).
    Distances are initialized to infinite in this function for assurance.
    Returns a graph RDD of the same form, only with correct distances.
    """
    # First initialize all of the distances
    dist_graph = graph.map(lambda (x, y): (x, (0, 'GRAY', y)) if (x == source_node) else (x, (MAX_DIST, 'BLACK', y)), preservesPartitioning = True)
    
    # Tell us when to stop... definitely don't want to stop before we begin
    keep_searching = True

    # Zero off our accumulator
    accum = sc.accumulator(0)

    # Cache our graph to avoid unnecessary computation due to laziness 
    dist_graph = dist_graph.cache()

    # Now map over the group, exploring one level each time
    iter_counter = sc.accumulator(0)
    while (keep_searching):
        # Keep a timer for useful output information
        before_combine = time.time()

        # Take a step from each node at the same time
        dist_graph = dist_graph.flatMap(explore_level_from_node)
        
        # Group results
        # This gives us an RDD of the form (node, [(d1, par1), (d2, par2), ...]
        dist_graph = dist_graph.groupByKey()
        
        # Now find the shortest distance, the lightest color, and reconstruct the adjacency list
        dist_graph = dist_graph.map(partial(fix_up_nodes, acc=accum))
        
        # Use a count to force evaluation of the map - this is needed to properly increment
        # the accumulator.
        dist_graph.count()

        print 'Time to combine on iteration number', iter_counter.value, ':', (time.time() - before_combine)

        # If we have gray nodes (i.e., ones to step from next round)
        # keep stepping. Otherwise, stop.
        if (accum.value == 0):
            keep_searching = False
        else:
            # Start the counter over
            accum = sc.accumulator(0)

        # Cache our graph to improve speed of future iterations
        dist_graph = dist_graph.cache()

        iter_counter.add(1)

    return dist_graph

def explore_level_from_node(node):
    """ Explores one level from the node node.
    Node is assumed to come in the form (node_name, (current_best_distance_to_source, curr_color, [neighbors]).
    Updates each node connected to this node to have distance +1.
    The update returns an empty adjacency list because we store every node in the whole graph again anyways.
    We store all nodes in the graph so as to not lose any BLACK or WHITE nodes.
    To be used with flatMap to parallel BFS from each node.
    The original (non-optimal) distances will be removed with the later fix_up_nodes.
    """

    # Just split up the data for readability
    curr_node = node[0]
    d = node[1][0]
    color = node[1][1]
    adj_list = node[1][2]

    # Our list of results to be returned returned for flatMap
    results = []

    # Iterate over all other nodes, update their distances, and return the parent.
    # Only do this if we have been to the node already - i.e., its color is GRAY. 
    # This is my equivalent of the rdd.filter() optimizations. When I used filter to only get the gray
    # nodes, it required a filter and then a union to put the white and black nodes back in. This took
    # significantly more time than just doing an if statement check like this.
    if (color == 'GRAY'):
        for other_node in adj_list:
            # Append our tentative new distance
            results.append((other_node, (d+1, 'GRAY', [])))

    # If this was black, we did NOT step from it, and we should save it just as it came in
    if (color == 'BLACK'):
        results.append(node)
    else:
        # Otherwise, it was either white or we stepped from it
        # If we stepped from it, we want it to be white, so:
        results.append((curr_node, (d, 'WHITE', adj_list)))

    return results

def fix_up_nodes(node, acc):
    """ After stepping from each node and grouping by key, we have a big RDD of the form
    (node, [(d1, color1, adj1), ...])
    We want to get the SHORTEST distance, the LIGHTEST color, and reconstruct the adjacency list.
    For every node we update to GRAY, we have another node to step from in the next round.
    This means we should increment our accumulator, which keeps track of how many we need to step from.
    """

    # Separate the two components of the tuple
    node_name = node[0]
    dist_node_list = node[1]

    # Extract the list of tuples into three separate lists
    dist_list = map(lambda (x, y, z): x, dist_node_list)
    color_list = map(lambda (x, y, z): y, dist_node_list)
    parents = map(lambda (x, y, z): z, dist_node_list)

    # The correct distance is the minimum distance
    actual_dist = min(dist_list)

    if 'WHITE' in color_list:
        # We've alreday been there, don't go again
        color = 'WHITE'
    elif 'GRAY' in color_list:
        # People for the next round
        # Also means we SHOULDN'T stop yet
        color = 'GRAY'
        acc.add(1)
    else:
        # Still haven't gotten to this guy
        color = 'BLACK'


    # Remove any duplicates and join lists of parents
    adj_list = list(set(reduce(lambda x, y: x + y, parents)))

    return (node_name, (actual_dist, color, adj_list))
