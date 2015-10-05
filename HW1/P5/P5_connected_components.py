from functools import partial
import time

def cc_2(sc, graph):
    """ Finds connected components of a symmetric graph.
    Graph is assumed to in the adjacency list representation of (node, (distance, [neighbors])).
    We assign a label to each graph, at the beginning just the name of the node.
    We then essentially BFS from all nodes at the same time, assigning a new label to each node given by
    the minimum label of all of its neighbors.
    We keep propagating this until no changes are made.
    At the end, the labels are the connected components.
    """
    
    # First initialize the labels and colors
    # Note that we start everyone off as WHITE - EVERYONE is active
    # Also note that everyone originally gets their name as a label
    # BLACK will denote inactive
    labeled_graph = graph.map(
            lambda (x, y): (x, (x, 'WHITE', y)),
            preservesPartitioning = True)
    
    # Tell us when to stop... definitely don't want to stop before we begin
    keep_searching = True

    # Zero off our accumulator - this will keep track of how many nodes were just updated
    # If a node's label was just updated, we need to propagate that information to its
    # neighbors, so clearly we cannot stop.
    accum = sc.accumulator(0)

    # Cache our graph to avoid unnecessary computation due to laziness 
    labeled_graph = labeled_graph.cache()

    # Now map over the group, exploring one level each time
    iter_counter = sc.accumulator(0)
    while (keep_searching):
        before_combine = time.time()

        # Take a step from each node at the same time and get new labels
        labeled_graph = labeled_graph.flatMap(update_labels).partitionBy(256)

        # Group results
        # This gives us an RDD of the form (node, [(label1, color1, par1), ...]
        # Despite a reduceByKey being faster in the BFS case, I found that a groupByKey followed
        # by a reduce was faster for finding connected components. I have been unable to explain why.
        labeled_graph = labeled_graph.groupByKey().partitionBy(256)
        
        # Reassign labels and colors correctly
        labeled_graph = labeled_graph.map(partial(get_correct_labels, acc=accum))
        
        # Use a count to force evaluation of the map - this is needed to properly increment
        # the accumulator.
        labeled_graph.count()

        print 'Time to combine on iteration number', iter_counter.value, ':', (time.time() - before_combine)

        # If we have gray nodes (i.e., ones to step from next round)
        # keep stepping. Otherwise, stop.
        if (accum.value == 0):
            keep_searching = False
        else:
            # Start the counter over
	    print 'Number of updated nodes:', accum.value, 'on iteration number:', iter_counter.value
            accum = sc.accumulator(0)

        # Cache our graph for improved speed on the next iteration
        labeled_graph = labeled_graph.cache()

        iter_counter.add(1)

    # Now that we are done, we need to find the number of connected components
    # and also the largest connected component
    labels = labeled_graph.map(
            lambda (node, (label, color, adj)): (label, 1)
            )

    # Combine counter to tally up the connected components
    counted_ccs = labels.reduceByKey(
            lambda counter1, counter2: counter1 + counter2
            )

    # Get the number of ccs
    num_ccs = counted_ccs.count()

    # And the largest cc
    # We take one sorted by 1 / the number of nodes, because sorting returns
    # in ascending order and we want the largest.
    # This will return [(label, count)] so we take [0][-1] to get count
    largest = counted_ccs.takeOrdered(1, key = lambda x: 1. / x[-1])[0][-1]

    return (num_ccs, largest)


def update_labels(node):
    """ Give the node's label to all adjacent nodes.
    This new label may or may not be the correct label for the adjacent nodes - this it to be determined
    in a later reduce.
    Node is assumed to come in the form (node_name, (current_label, current_color, [neighbors]).
    The update returns an empty adjacency list because we store every node in the whole graph again anyways.
    We store all nodes in the graph so as to not lose any BLACK or WHITE nodes.
    To be used with flatMap to parallel BFS from each node.
    The original (non-optimal) distances will be removed with the later fix_up_nodes.
    """

    # Just split up the data for readability
    curr_node = node[0]
    label = node[1][0]
    color = node[1][1]
    adj_list = node[1][2]

    # Our list of results to be returned returned for flatMap
    results = []

    # Iterate over all other nodes, update their labels
    # Only do this if the node has information to propagate - i.e., it is WHITE
    # This is my equivalent of the rdd.filter() optimizations. When I used filter to only get the WHITe 
    # nodes, it required a filter and then a union to put the BLACK back in. This took
    # significantly more time than just doing an if statement check like this.
    if (color == 'WHITE'):
        for other_node in adj_list:
            # update our tentative new label 
            # Alter the color of the node to WHITE - it might now has information to give
            results.append(
			    (other_node, (label, 'WHITE', []))
			  )

    # We append the node so that we do not lose any of the inactive nodes
    # Note that we have now sent this nodes information (or it did not have any to begin with) - it should now become inactive!
    results.append(
            (curr_node, (label, 'BLACK', adj_list))
            )

    return results

def get_correct_labels(node, acc):
    """ After getting new labels from each node and grouping by key, we have a big RDD of the form
    (node, [(label1, color1, adj1), ...])
    We want to get the smallest label and reconstruct the adjacency list.
    If the distance was updated - i.e., if the distance came from a WHITE node (not yourself),
    then we need to continue propagating that information.
    If the distance came from a BLACK node - i.e., yourself - the distance was not updated and this node
    can be considered inactive.
    We increment the accumulator accordingly.
    """

    # Just separate the name and the tuple of distances, paths,
    # colors, and adjacency lists
    node_name = node[0]
    label_node_list = node[1]

    # Get the distances, paths, colors, and potential adjacency lists all in list form
    label_list = map(lambda (x, y, z): x, label_node_list)
    color_list = map(lambda (x, y, z): y, label_node_list)
    potential_adj_list = map(lambda (x, y, z): z, label_node_list)

    # The correct label is the minimum label
    actual_label = min(label_list)

    #### NOTE: 'BLACK' should clearly be in ALL lists, as we map across all nodes
    #### and even if we do not propagate any information, return the node with the label 'BLACK'.
    #### This algorithm works perfectly on the marvel data but I was getting an error with the Wiki
    #### dataset that 'BLACK' was not in a certain node's list.
    #### For that reason I added this if statement. It seems to imply that some page links to a page
    #### that is not actually in the dataset?
    if 'BLACK' in color_list:
        # The only node in the list colored BLACK is this node
        # That will tell us what its old label was
        my_index = color_list.index('BLACK')

        # If we haven't been updated, we can't contribute to the update right now
        if label_list[my_index] == actual_label:
            color = 'BLACK'
        # But if we have been updated, we need to send that info along
        else:
            color = 'WHITE'
            acc.add(1)

        # Add all the adjacency lists from all the steps and get rid of any duplicates
        # This actually might not be necessary, but we will do so for safety
        try:
            adj_list = list(set(reduce(lambda x, y: x + y, potential_adj_list)))
        except:
            print potential_adj_list

        return (node_name, (actual_label, color, adj_list))
    else:
        #### NOTE: otherwise, we are on that one strange node causing an error...
        adj_list = list(set(reduce(lambda x, y: x + y, potential_adj_list)))

        # We might as well propagate it's label...
        color = 'WHITE'
        
        # Log some information about what's going on here...
        with open('error.log', 'a') as log_file:
            print node_name, 'is not in the original dataset?'
            print >> log_file, node_name, actual_label, adj_list 

        # Just reappend the node (as it should be there anyways) and carry on
        return (node_name, (actual_label, color, adj_list))

def get_correct_labels_reduce_func(label_node_list1, label_node_list2, acc):

    label1 = label_node_list1[0]
    c1 = label_node_list1[1]
    adj1 = label_node_list1[2]

    label2 = label_node_list2[0]
    c2 = label_node_list2[1]
    adj2 = label_node_list2[2]

    c = 'BLACK'
    label = label1

    if c1 == 'BLACK':
        # Then this is the OLD distance
        if label1 <= label2:
            # Then we shouldnt be updated and we have no new info
            c = 'BLACK'
            label = label1
        else:
            # We got updated - update the label and set node for propagation
            # Note - at SOME POINT we will have to compare our old label to a new one
            # We may even end up finding a BETTER label after this comparison, but that's fine
            # the point is that the label still changed
            c = 'WHITE'
            label = label2
            acc.add(1)
    elif c2 == 'BLACK':
        # Symmetric case
        if label2 <= label1:
            c = 'BLACK'
            label = label2
        else:
            c = 'WHITE'
            label = label1
            acc.add(1)
    else:
        # If neither of them are black, just take the better one
        c = 'WHITE'
        label = min(label1, label2)

    # Reconstruct the adjacency list and (maybe) remove duplicates
    adj_list = list(set(adj1 + adj2))

    return (label, c, adj_list)
