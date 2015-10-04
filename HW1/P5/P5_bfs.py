def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner

def bfs_parents(graph, start, end):
    context = graph.context
    iteration = context.accumulator(0)
    data = context.emptyRDD()
    toVisit = graph.filter(lambda KV: KV[0] == start).mapValues(lambda v: (v, ''))
    while not toVisit.isEmpty():
        distance = iteration.value
        new_data = toVisit.mapValues(lambda v: (distance, v[1]))
        data = data.union(new_data).reduceByKey(lambda x, y: min(x, y)).partitionBy(24)
        if len(data.lookup(end)) > 0:
            return data
        neighbors = toVisit.map(lambda KV: [(x, KV[0]) for x in KV[1][0]]).flatMap(lambda KV: KV).partitionBy(24)
        assert copartitioned(graph, neighbors)
        toVisitPlus = graph.join(neighbors)
        assert copartitioned(toVisitPlus, data)
        toVisit = toVisitPlus.subtractByKey(data).cache()
        assert copartitioned(graph, toVisit)
        iteration.add(1)
    return data
    
def shortest_path(data, end):
    sorted_data = data.sortByKey().cache()
    path = []
    next = end
    while next != '':
        node = sorted_data.lookup(next)
        if len(node) > 0:
            path.append((next, node[0][0]))
            next = node[0][1]
            if len(node) > 1:
                path.append('ERROR: NON-UNIQUE RESULT')
        else:
            next = ''
    path.reverse()
    return path
