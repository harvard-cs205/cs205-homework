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
def ssbfs(start_name, graph, sc):
    print "Running SS-BFS for ", start_name
    nodes = sc.parallelize([(start_name, 0)])
    itr_acc = sc.accumulator(0)
    total_acc = sc.accumulator(1)
    while True:
        itr_acc+=1
        i = itr_acc.value
        new_nodes = nodes.filter(lambda node: filterer(node[1] + 1)).join(graph).flatMap(lambda node: node[1][1]).distinct().map(lambda x: (x, i))
        # Cogroup the input nodes with the new nodes, taking only the first iteration of each character as its value
        nodes = nodes.cogroup(new_nodes).map(lambda x: getCogrouper(x)).partitionBy(8).cache()
        node_count = nodes.count()
        
        # print "NODES PARTITIONS ", nodes.partitioner.numPartitions
        # print "GRAPH PARTITIONS ", graph.partitioner.numPartitions
        # Short-circuit if we didn't add any new nodes this iteration
        if node_count == total_acc.value:
            break
        print "Characters added in iteration %s: %s"  % (itr_acc.value, node_count - total_acc.value)
        total_acc+=node_count - total_acc.value
    print  "Characters connected to %s: %s" % (start_name, nodes.count())
    return nodes.count()