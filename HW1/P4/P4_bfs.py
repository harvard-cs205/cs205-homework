def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner

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

#######
# Implements Breadth First Search
# Arguments:
# 	adj (KV RDD): Adjacency list graph representation. Value is an iterator over neighbors.
# 	start (string): Where the breadth first search will start
#   sc: SparkContext
# Returns:
# 	(RDD) Distances to each point
def bfs(adj, start, sc, numPartitions):
    accum = sc.accumulator(0)
    adj = adj.map(lambda x: x).partitionBy(numPartitions).cache()


    distances = adj.mapValues(lambda _: -1).partitionBy(numPartitions)
    traversed = sc.parallelize([(start, 0)]).cache().partitionBy(numPartitions)
    distances = distances.fullOuterJoin(traversed).mapValues(lambda (x, y): x if y == None else y).partitionBy(numPartitions).cache()
    
    assert(copartitioned(adj, distances))
    farthest = 0
    accum.add(1)
    while accum.value != 0:
        accum.value = 0
        print "\n\nBFS: On iteration ", farthest, ' for ', start, '\n\n'

        # Get the nodes that are farthest away
        farthest_nodes = distances.filter(lambda (node, dist): dist == farthest)

        # Get their neighbors, and get the distances of each neighbor
        assert(copartitioned(farthest_nodes,adj))
        joined_farthest_neighbors = farthest_nodes.join(adj, numPartitions)
        neighbor_distances = joined_farthest_neighbors.values().flatMap(divider).partitionBy(numPartitions).cache()
        
        # Combine the distances of the neighbors with the distances we already have
        assert(copartitioned(neighbor_distances, distances))
        distances = distances.cogroup(neighbor_distances).mapValues(reducer).cache()
        farthest += 1

        # For each element just added, increment the accumulator
        distances.filter(lambda (x, dist): dist == farthest).foreach(lambda x: accum.add(1))

    return distances

