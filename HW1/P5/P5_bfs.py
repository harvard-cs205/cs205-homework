def bfs_parents(graph, start):
    context = graph.context
    iteration = context.accumulator(0)
    data = context.emptyRDD()
    toVisit = graph.filter(lambda KV: KV[0] == start).map(lambda KV: (KV[0], (KV[1], '')))
    while not toVisit.isEmpty():
        distance = iteration.value
        new_data = toVisit.mapValues(lambda v: (distance, v[1]))
        data = data.union(new_data).reduceByKey(lambda x, y: min(x, y)).partitionBy(24)
        pairs = toVisit.map(lambda KV: [(x, KV[0]) for x in KV[1][0]])
        neighbors = pairs.flatMap(lambda KV: KV).partitionBy(24)
        toVisit = graph.join(neighbors).subtractByKey(data)
        toVisit.cache()
        iteration.add(1)
    return data
    
def shortest_path(data, end):
    sorted_data = data.sortByKey()
    sorted_data.cache()
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
