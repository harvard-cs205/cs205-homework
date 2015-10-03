def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner

#######
# Implements Breadth First Search
# Arguments:
# 	adj (KV RDD): Edge list. For example: [(1,2), (1, 3), (2,3), (3, 2)] is the graph where there is a directed edge from 1 to 2 and 1 to 3, edges in both directions between 2 and 3.
# 	start (string): Where the breadth first search will start
#   If a stopNode is provided, then it will stop iterating once that node is found.
# Returns:
# 	(RDD) Distances to each point, and (RDD) list of all points that were not reachable.
def bfs(adj, start, sc, numPartitions, list_of_nodes, stopNode=None):
    accum = sc.accumulator(0)
    print 'Adjacency partitions:', adj.getNumPartitions()
    adj = adj.map(lambda x: x).partitionBy(numPartitions).cache()
    def reducer(tup):
        left_val = tup[0]
        right_vals = tup[1]
        if len(right_vals) == 0:
            return list(left_val)[0]
        elif list(left_val)[0] == -1:
            return list(right_vals)[0]
        else:
            return list(left_val)[0]
    
    def divider(tup):
        dist = tup[0]
        node_vals = tup[1]
        return [(nv, dist + 1) for nv in node_vals]


    distances = adj.mapValues(lambda _: -1)

    traversed = sc.parallelize([(start, 0)]).cache().partitionBy(numPartitions)
    distances = distances.fullOuterJoin(traversed).mapValues(lambda (x, y): x if y == None else y).partitionBy(numPartitions).cache()
    #print distances.take(100)
    assert(copartitioned(adj, distances))
    print 'Start traverse partitions:', distances.getNumPartitions()
    farthest = 0
    accum.add(1)
    while accum.value != 0:
        accum.value = 0
        print "\n\nOn iteration ", farthest, ' for ', start, '\n\n'
        if stopNode != None and distances.lookup(stopNode)[0] != -1:
            break
        farthest_nodes = distances.filter(lambda (node, dist): dist == farthest)

        joined_farthest_neighbors = farthest_nodes.join(adj, numPartitions)
        neighbor_distances = joined_farthest_neighbors.values().flatMap(divider).partitionBy(numPartitions).cache()
        assert(copartitioned(neighbor_distances, distances))
        distances = distances.cogroup(neighbor_distances).mapValues(reducer).cache()
        farthest += 1
        distances.filter(lambda (x, dist): dist == farthest).foreach(lambda x: accum.add(1))
        #print traversed.take(100)

    print 'End tranverse partitions:', distances.getNumPartitions()
    return distances, distances.filter(lambda (node, dist): dist < 0)