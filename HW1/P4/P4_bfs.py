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

def shortest_path_parallel(graph, root_node):
    depth = 0
    queue = set([root_node]) # set to store nodes that will be need to be visited
#     print queue
#     print len(queue)
    traversed_nodes = set() # set to store traversed nodes and its distance from root_node
    
    while(len(queue) > 0): # while there are more nodes in queue to be visisted
#         print depth
        depth += 1 # increment the depth
        # filter on only the nodes in the queue
        # grab out their children
        filtered_graph = graph.filter(lambda (Node, V): Node in queue).collect()
        # update the traversed nodes with the nodes in the queue
        traversed_nodes.update(queue)
        # to store the children of traversed nodes from the queue
        traversed_nodes_children = set()
        
        # Loop through each node in the filtered_graph to add on children
        for node in filtered_graph:
            # children of each node
            children = set(node[1])
            # update the children to the traversed nodes
            traversed_nodes_children.update(children)
        # update queue by taking
        queue = traversed_nodes_children - traversed_nodes
    return len(traversed_nodes) - 1, depth
            
        
