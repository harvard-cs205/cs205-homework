def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner

def connected_components(graph):
    context = graph.context
    result = []
    while not graph.isEmpty():
        visited = context.emptyRDD()
        start = graph.first()
        toVisit = graph.filter(lambda KV: KV == start)
        while not toVisit.isEmpty():
            #Update record of previously visited nodes
            assert copartitioned(graph, toVisit)
            graph = graph.subtract(toVisit).partitionBy(16).cache()
            visited = visited.union(toVisit).partitionBy(16)
            #Update record of nodes to visit
            neighbors = toVisit.values().flatMap(lambda x: x).collect()
            toVisit = graph.filter(lambda KV: KV[0] in neighbors).cache()
            assert copartitioned(graph, toVisit)
        result.append(visited.count())
    return result
