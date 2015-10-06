from P5_bfs import *
import pyspark

#####################################################################################
#
# STATEMENT AT FIRST:
# Well, honestly this approach is really naive and invloves little parallel, at least
# it passed all tests in my test dataset, and as long as the memory of single machine
# is enough to store the graph, the excecution time is under control
#
#####################################################################################

sc = pyspark.SparkContext()

#####################################################################################
# The function to look for connected components.
# The worst case scenario for computation time would be when all nodes are connected together
# and it will take n * (n-1) time to converge
#####################################################################################

def connected_components(graph):
# List of connected components found. The order is random.
    result = []
    # Make a copy of the set, so we can modify it.
    nodes = set(graph)
    # Iterate while we still have nodes to process.
    while nodes:
        # Get a random node and remove it from the global set.
        n = nodes.pop()
        # This set will contain the next group of nodes connected to each other.
        group = {n}
        # Build a queue with this node in it.
        queue = [n]
        # Iterate the queue.
        # When it's empty, we finished visiting a group of connected nodes.
        while queue:
            # Consume the next item from the queue.
            n = queue.pop(0)
            # Fetch the neighbors.
            neighbors = n[1][1]
            # Remove the neighbors we already visited.
            neighbors.difference_update(group)
            # Remove the remaining nodes from the global set.
            nodes.difference_update(neighbors)
            # Add them to the group of connected nodes.
            group.update(neighbors)
            # Add them to the queue, so we visit them in the next iterations.
            queue.extend(neighbors)

        # Add the group to the list of groups.
        result.append(group)

    # Return the list of groups.
    return result

if __name__ == '__main__':

    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
    page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)

    symmetric_graph = links.map(construct_neighbor_graph).collect()
    components = connected_components(symmetric_graph)

    print "The number of connected components is ", len(components), '.\n'
    max = 0

    for i in xrange(len(components)):
        print 'The number of nodes in component', i, ' is', len(components[i])
        if len(components[i]) > max: max = len(components[i])

    print 'The number of nodes in the largest connected component is', max

