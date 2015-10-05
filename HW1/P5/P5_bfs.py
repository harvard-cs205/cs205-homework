from functools import partial
import time

MAX_DIST = 10**16

def BFS(graph, source_node, target_node, sc):
    """ Runs a parallel BFS on graph.
    Graph is assumed to in the adjacency list representation of (node, (distance, [neighbors])).
    Distances are initilized to infinite in this function for assurance.
    Returns a graph RDD of the same form, only with correct distances.
    """
    # First initialize all of the distances, colors, and path lists
    dist_graph = graph.map(lambda (x, y): (x, (0, [], 'GRAY', y)) if (x == source_node) else (x, (MAX_DIST, [], 'BLACK', y)), preservesPartitioning = True)
    
    # Tell us when to stop... definitely don't want to stop before we begin
    keep_searching = True

    # Zero off our accumulator
    accum = sc.accumulator(0)

    # Cache our graph to avoid unnecessary computation due to laziness 
    dist_graph = dist_graph.cache()

    # Arbitrarily set the target distance to a large number
    # Also give us an array to hold the target path
    target_d = 100000
    target_path = []

    # Now map over the group, exploring one level each time
    iter_counter = sc.accumulator(0)
    while (keep_searching):
        before_combine = time.time()

        # Take a step from each node at the same time
        # We range partition our graph so that the following reduce has no shuffle at all
        dist_graph = dist_graph.flatMap(explore_level_from_node).partitionBy(256)
        
        # Reduce by key with the reduce function below to get the best distances, the correct path,
        # and the lightest color. Originally implemented with a groupByKey and then a reduce,
        # but found this to be faster due to reduction of shuffles
        dist_graph = dist_graph.reduceByKey(partial(fix_up_nodes, acc=accum))

        # Use a count to force evaluation of the map - this is needed to properly increment
        # the accumulator.
        dist_graph.count()

        # See if we have found the target node yet
        target = dist_graph.lookup(target_node)[0]

        # If we found our target
        if target[2] != 'BLACK':
            target_d = target[0]
            target_path = target[1] + [target_node]
            keep_searching = False

        # Print out some useful timing information
        print 'Time to combine on iter', iter_counter.value, ':', (time.time() - before_combine)

        # If we have gray nodes (i.e., ones to step from next round)
        # keep stepping. Otherwise, stop.
        if (accum.value == 0):
            keep_searching = False
        else:
            # Start the counter over
	    print 'Number of updated nodes:', accum.value, 'on iteration number:', iter_counter.value
            accum = sc.accumulator(0)

        # Cache our graph for improved speed on the next iteration
        dist_graph = dist_graph.cache()

        iter_counter.add(1)

    return (target_d, target_path)

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
    path = node[1][1]
    color = node[1][2]
    adj_list = node[1][3]

    # Our list of results to be returned returned for flatMap
    results = []

    # Iterate over all other nodes, update their distances, and return the parent.
    # Only do this if we have been to the node already - i.e., its color is GRAY 
    # This is my equivalent of the rdd.filter() optimizations. When I used filter to only get the gray
    # nodes, it required a filter and then a union to put the white and black nodes back in. This took
    # significantly more time than just doing an if statement check like this.
    if (color == 'GRAY'):
        # Update the path: how we get here is how we got to node, plus node!
        new_path =  path + [curr_node]
        for other_node in adj_list:
            # Append our tentative new distance
            # Alter the color to Gray
            results.append(
			    (other_node, (d+1, new_path, 'GRAY', []))
			  )

    # Now lets not lose any nodes
    # If this was black, we did NOT step from it, and we should save it just as it came in
    if (color == 'BLACK'):
        results.append(node)
    else:
        # Otherwise, it was either white or we stepped from it
        # If we stepped from it, we want it to be white, so:
        results.append((curr_node, (d, path, 'WHITE', adj_list)))

    return results

def fix_up_nodes(dist_node_list1, dist_node_list2, acc):
    """ Reduce function to be used after flatMap explore_level_from_node.
    Gets the best distance, the path, the correct color, and the adjacency list.
    """

    # Just get the variables from the two lists
    d1 = dist_node_list1[0]
    p1 = dist_node_list1[1]
    c1 = dist_node_list1[2]
    par1 = dist_node_list1[3]
    d2 = dist_node_list2[0]
    p2 = dist_node_list2[1]
    c2 = dist_node_list2[2]
    par2 = dist_node_list2[3]

    # The correct distance is the minimum
    d = min(d1, d2)

    # Pick the path
    # A few edge cases: if neither is the empty list, we take the shorter one
    # If one of them IS the empty list, we want the OTHER one
    p = []
    if (p1 != []) and (p2 != []):
        len1 = len(p1)
        len2 =len(p2)
        p = p1
        if len2 < len1:
            p = p2
    else:
        if p1 == []:
            p = p2
        else:
            p = p1

    c = 'BLACK'
    # We just want the lightest color
    if 'WHITE' in [c1, c2]:
        c = 'WHITE'
    elif 'GRAY' in [c1, c2]:
        c = 'GRAY'
        acc.add(1)

    # And the adjacency list comes from reconstruction
    adj_list = list(set(par1 + par2))

    return (d, p, c, adj_list)
