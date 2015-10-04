def lesserFirst(x):
    tmp_list = combineLists([x[0]], x[1])
    least = float("inf")
    for i in tmp_list:
        if i < least:
            least = i
    return (least, tmp_list)

def combineLists(x,y):
    new_list = x
    for yi in y:
        if yi not in new_list:
            new_list = new_list + [yi]
    return new_list
     
def combinePotentialLists(x,y):
    new_list = []
    if len(x) > 0:
        new_list = x
    if len(y) > 0:
        for yi in y:
            if yi not in new_list:
                new_list = new_list + [yi]
    return new_list

def takeOnlyDoubles(x,y):
    if x in y:
        return [x]
    else:
        return []
    
def filterer(i):
    def filt(x):
        return x==itr_acc.value
    return filt

# Build a tuple with the name and the first element in the list of cogrouped terms (the iteration id) thats not empty
def getCogrouper(x):
    for item in list(x[1]):
        item = list(item)
        if len(item) > 0:
            return (x[0], item[0])

# Single-Source Breadth-First Search using no collects :)
def ssbfs(start_name, graph, sc, num_partitions):
    print "Running SS-BFS for key ", start_name
    nodes = sc.parallelize([(start_name, 0)])
    itr_acc = sc.accumulator(0)
    total_acc = sc.accumulator(1)
    while True:
        itr_acc+=1
        i = itr_acc.value
        new_nodes = nodes.filter(lambda node: filterer(node[1] + 1)).join(graph).flatMap(lambda node: node[1][1]).distinct().map(lambda x: (x, i))
        # Cogroup the input nodes with the new nodes, taking only the first iteration of each character as its value
        nodes = nodes.cogroup(new_nodes).map(lambda x: getCogrouper(x)).partitionBy(num_partitions).cache()
        node_count = nodes.count()

        # Short-circuit if we didn't add any new nodes this iteration
        if node_count == total_acc.value:
            break
        print "Nodes added in iteration %s: %s"  % (itr_acc.value, node_count - total_acc.value)
        total_acc+=node_count - total_acc.value
    #print  "Nodes connected to %s: %s" % (start_name, nodes.count())
    return nodes

def connectedComponents(page_links, sc, num_partitions):
    symmetric_connected_node_list = page_links.map(lambda x: lesserFirst(x)).reduceByKey(lambda x,y: combineLists(x,y))
    symmetric_connected_node_list = symmetric_connected_node_list.union(symmetric_connected_node_list.flatMapValues(lambda x: x).map(lambda x: (x[1], [x[0]])))    
    nodes_left = symmetric_connected_node_list
    
    num_groups = 0
    largest_size = 0
    total_num_nodes_touched = 0

    while True: 
        print '----------------------'
        print "New Iteration - starting with %s nodes left to collect" % nodes_left.count()
        current = nodes_left.take(1)
        current = current[0][0]
        linked_nodes = ssbfs(current, symmetric_connected_node_list, sc, num_partitions)
        num_groups+=1
        current_group_count = linked_nodes.count()
        if (current_group_count) > largest_size:
            largest_size = current_group_count
        total_num_nodes_touched += current_group_count
        
        nodes_left = nodes_left.subtractByKey(linked_nodes).partitionBy(num_partitions).cache()
        
        if nodes_left.count() == 0:
            print "No reduction! Breaking!"
            break

    print '----------------------'
    print "Number of Connected Components: ", num_groups
    print "Largest Component num nodes: ", largest_size
    print "Total number of nodes touched: ", total_num_nodes_touched
    return num_groups
    
# Counts as an edge even if the link goes in only one direction
def uniConnectedComponents(page_links, sc, num_partitions):
    print "Running Uni-Directional Connected Components"
    return connectedComponents(page_links, sc, num_partitions)
    
# Only counts as an edge if the link goes in both directions
def biConnectedComponents(page_links, sc, num_partitions):
    print "Running Bi-Directional Connected Components"
    selves = page_links.keys().map(lambda x: (x, [x]))
    page_links = page_links.union(selves)
    
    symmetric_connected_node_list = page_links
    num_ccs = symmetric_connected_node_list.count()

    symmetric_connected_nodes = symmetric_connected_node_list.flatMapValues(lambda x: x)
    symmetric_connected_nodes_r = symmetric_connected_nodes.map(lambda x: (x[1], x[0]))
    nodes = symmetric_connected_nodes_r.join(symmetric_connected_node_list)
    
    nodes = nodes.map(lambda x: (x[0], takeOnlyDoubles(x[1][0], list(x[1][1]))))
    nodes = nodes.map(lambda x: lesserFirst(x)).reduceByKey(lambda x,y: combineLists(x,y))
    nodes = connectedComponents(nodes, sc, num_partitions)
    return nodes