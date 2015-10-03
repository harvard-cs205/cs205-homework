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
def bfs(adj, start, sc, numPartitions, distances=None, stopNode=None):
    accum = sc.accumulator(0)
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

    if distances == None:
        distances = adj.mapValues(lambda _: -1)

    traversed = sc.parallelize([(start, 0)]).cache().partitionBy(numPartitions)
    distances = distances.fullOuterJoin(traversed).mapValues(lambda (x, y): x if y == None else y).partitionBy(numPartitions).cache()
    #print distances.take(100)
    assert(copartitioned(adj, distances))
    farthest = 0
    accum.add(1)
    while accum.value != 0:
        accum.value = 0
        print "\n\nBFS: On iteration ", farthest, ' for ', start, '\n\n'
        if stopNode != None and distances.lookup(stopNode)[0] != -1:
            break
        farthest_nodes = distances.filter(lambda (node, dist): dist == farthest)

        joined_farthest_neighbors = farthest_nodes.join(adj, numPartitions)
        neighbor_distances = joined_farthest_neighbors.values().flatMap(divider).partitionBy(numPartitions).cache()
        assert(copartitioned(neighbor_distances, distances))
        distances = distances.cogroup(neighbor_distances).mapValues(reducer).cache()
        farthest += 1
        distances.filter(lambda (x, dist): dist == farthest).foreach(lambda x: accum.add(1))

    return distances, distances.filter(lambda (node, dist): dist < 0)

## This is for the connected components
# Approach: BFS until it quits. Then reduce the distances to only the ones that are unreachable. Increment counter by 1. Restart BFS with the new set of distances.
def count_connected_components(adj, numPartitions, sc):
    adj = adj.map(lambda x: x).partitionBy(numPartitions).cache()
    dists = adj.mapValues(lambda _: -1)
    num_left = sc.accumulator(0)
    dists.foreach(lambda x: num_left.add(1))
    num_conn_components = 0
    while(num_left.value > 0):
        num_conn_components += 1
        num_left.value = 0
        start = dists.takeSample(True, 1)[0][0]
        dist, unreachable = bfs(adj, start, sc, numPartitions, distances=dists, stopNode=None)
        dists = dist.filter(lambda (node, dist): dist < 0)
        dists.foreach(lambda x: num_left.add(1))
        print "CONNECTED_COMPONENTS: Number of elements not yet reached: ", num_left.value
        
    return num_conn_components