import findspark; findspark.init()
import pyspark
import matplotlib.pyplot as plt
import seaborn as sns

def shortest_path(graph, root_node, iteration):
    queue = [root_node]
    traversed_nodes = {} # dict to store traversed nodes and its distance from root_node
    # For loop through iterations
    for i in xrange(0, iteration+1):
        neighbor = []
        # Loop through each node in the queue
        for node in queue:
            # When we encounter a new node
            if node not in traversed_nodes:
                # Node is new and set the distance
                traversed_nodes[node] = i
                # Add in the new node's neighbors
#                 print graph.lookup(node)
                # Add the new node's neighboring nodes to adj neighbors
                neighbor = neighbor + graph.lookup(node)[0]
        # Refresh queue
        queue = neighbor
    # Result
    result = graph.map(lambda (k,v): (k, v, traversed_nodes[k]) if k in traversed_nodes else (k, v, -1))
    # Number of nodes touched excluding the root note
    num_nodes = len(traversed_nodes) - 1 # -1 for excluding the root node
    print "Num of nodes touched: ", num_nodes
    return num_nodes, result
