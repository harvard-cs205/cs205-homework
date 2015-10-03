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
    solutions = sc.accumulator(0)
    adj = adj.map(lambda x: x).partitionBy(numPartitions).cache()
    to_search = adj.filter(lambda (x, y): x == start)
    paths = to_search.map(lambda (x, neighbors): (x, [x]))
    #res = paths


    #elem looks like (node_to_Search_from, (([neighbors], _), [path]))
    #From here we want to emit all of the paths that this generates
    def reducingFun(elem, start, end, counter):
        (prev, ((neighbors, _), path)) = elem
        generated_paths = []
        for neighbor in neighbors:
            generated_paths.append((neighbor, path + [neighbor]))
            if end == neighbor:
                counter.append(1)
        return generated_paths

    def not_found(counter):
        counter.value == 0

    while solutions.value == 0:
        #while True:
        print "got here"
        searchers = adj.join(to_search)
        joined = searchers.join(paths)
        print joined.take(5)
        #return None, None

        # THis looks something like (node, [path]), (node, [path]) ....
        paths = joined.flatMap(lambda x: reducingFun(x, start, stopNode, solutions))

        # Get the outlying nodes.
        to_search = paths.keys().distinct().map(lambda x: (x, [-1]))
        print paths.take(10)

    res = paths.filter(lambda (x, y): x == stopNode).values()
    return res
    
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